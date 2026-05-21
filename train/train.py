from pathlib import Path
import re
import os
import itertools
from typing import Tuple
from contextlib import ExitStack, contextmanager

from beartype import beartype

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchgen.gen_aoti_c_shim import base_type_to_c_type

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import WMEmbedder, WMDetector
from dataset.audio_dataset import get_dataloader, audioDataset
from optimizer import get_optimizer
from torch.utils import tensorboard
from losses import (
    adversarial_loss_d,
    adversarial_loss_g,
    bits_to_chunks,
    cos_loss,
    decoding_loss,
    feature_loss,
    mel_loss,
    vad_based_loss,
)
import json
from STmodels.model import SpeechTokenizer
import time
from tqdm import tqdm
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs, DataLoaderConfiguration
from attacks.augmentation import attack_augmentation, ATTACK_REGISTRY
import datetime
import random
from collections import defaultdict
from quality import compute_quality




# helpers

def exists(val):
    return val is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def checkpoint_num_steps(checkpoint_path):
    """Returns the number of steps trained from a checkpoint based on the filename.

    Filename format assumed to be something like "/path/to/soundstorm.20000.pt" which is
    for 20k train steps. Returns 20000 in that case.
    """
    results = re.findall(r'\d+', str(checkpoint_path))

    if len(results) == 0:
        return 0

    return int(results[-1])


class WMTrainer(nn.Module):
    @beartype
    def __init__(
        self,
        generator: SpeechTokenizer,
        discriminators: dict,
        cfg,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()
        torch.manual_seed(cfg.get('seed'))

        base_results_folder = Path(cfg.get('results_folder'))

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        dataloader_config = DataLoaderConfiguration(split_batches=cfg.get("split_batches", False))
        self.accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs
        )

        # Create a unique run directory with timestamp to avoid conflicts
        run_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.results_folder = base_results_folder / run_name

        # Only main process writes config and creates TensorBoard writer
        if self.is_main:
            self.results_folder.mkdir(parents=True, exist_ok=True)
            with open(self.results_folder / 'config.json', 'w+') as f:
                json.dump(cfg, f, ensure_ascii=False, indent=4)
            log_dir = os.path.join(self.results_folder, 'logs')
            self.writer = tensorboard.SummaryWriter(log_dir)
        else:
            self.results_folder.mkdir(parents=True, exist_ok=True)
        self.generator = generator
        self.discriminators = discriminators
        self.batch_size = cfg.get("batch_size")
        self.epochs = cfg.get("epochs")
        # lr
        self.lr = cfg.get("learning_rate")
        self.initial_lr = cfg.get("intial_learning_rate")

        self.num_warmup_steps = cfg.get("num_warmup_steps")
        self.steps = torch.Tensor([0])
        self.best_dev_loss = float('inf')
        self.sample_rate = cfg.get('sample_rate')
        self.showpiece_num = cfg.get('showpiece_num', 8)
        self.save_model_steps = cfg.get('save_model_steps')
        self.vad_loss_lambda = cfg.get('vad_loss_lambda')
        self.cos_loss_lambda = cfg.get('cos_loss_lambda')
        self.mel_loss_lambda = cfg.get('mel_loss_lambda')
        self.adv_loss_lambda = cfg.get('adv_loss_lambda')
        self.dec_loss_lambda = cfg.get('dec_loss_lambda')
        self.multi_scale_mel_loss_lambdas = cfg.get('multi_scale_mel_loss_lambdas')
        self.multi_scale_mel_loss_kwargs_list = []
        mult = 1
        for i in range(len(self.multi_scale_mel_loss_lambdas)):
            self.multi_scale_mel_loss_kwargs_list.append({'n_fft': cfg.get('n_fft') // mult,
                                                           'num_mels': cfg.get('num_mels'), 
                                                           'sample_rate': self.sample_rate,
                                                           'hop_size': cfg.get('hop_size') // mult, 
                                                           'win_size': cfg.get('win_size') // mult, 
                                                           'fmin': cfg.get('fmin'),
                                                           'fmax': cfg.get('fmax')})
            mult = mult * 2
        self.mel_kwargs = {'n_fft': cfg.get('n_fft'), 
                           'num_mels': cfg.get('num_mels'), 
                           'sample_rate': self.sample_rate,
                           'hop_size': cfg.get('hop_size'), 
                           'win_size': cfg.get('win_size'), 
                           'fmin': cfg.get('fmin'),
                           'fmax': cfg.get('fmax')}

        self.msg_processor = WMEmbedder(
            nbits=16,
            input_dim=1024,
            nchunk_size=4,
        ).to(self.device)
        self.detector = WMDetector(
            1024,
            16,
            nchunk_size=4,
        ).to(self.device)

        # dataset
        with open(cfg.get("train_files"), 'r') as f:
            train_files = f.readlines()
        with open(cfg.get("valid_files"), 'r') as f:
            valid_files = f.readlines()

        self.ds = audioDataset(file_list=train_files,
                               segment_size=cfg.get("segment_size"),
                               downsample_rate=generator.downsample_rate,
                               sample_rate=self.sample_rate)
        self.valid_ds = audioDataset(file_list=valid_files,
                                     segment_size=cfg.get("segment_size"),
                                     downsample_rate=generator.downsample_rate,
                                     sample_rate=self.sample_rate,
                                     valid=True)

        self.dl = get_dataloader(self.ds, batch_size=self.batch_size, shuffle=True,
                                 drop_last=cfg.get("drop_last", True), num_workers=cfg.get("num_workers"))
        self.valid_dl = get_dataloader(self.valid_ds, batch_size=1, shuffle=False, drop_last=False, num_workers=1)

        # optimizers
        # train msg_processor and detector only
        self.trainable_params = list(self.msg_processor.parameters()) + list(self.detector.parameters())
        self.optim_generator = get_optimizer(
            self.trainable_params,
            lr=self.lr,
            wd=cfg.get("wd"),
            betas=cfg.get("betas")
        )
        self.optim_discriminators = get_optimizer(itertools.chain(*[d.parameters() for d in discriminators.values()]),
                                                  lr=self.lr, wd=cfg.get("wd"), betas=cfg.get("betas"))

        # scheduler
        num_train_steps = self.epochs * len(self.ds) // self.batch_size
        self.scheduler_generator = CosineAnnealingLR(self.optim_generator, T_max=num_train_steps)
        self.scheduler_discriminator = CosineAnnealingLR(self.optim_discriminators, T_max=num_train_steps)

        # Only wrap trainable models in DDP. Generator is frozen — no gradient sync needed.
        # Wrapping the ~33M param frozen generator in DDP wastes massive communication bandwidth.
        self.generator = self.generator.to(self.device)
        for param in self.generator.parameters():
            param.requires_grad = False

        self.msg_processor, self.detector, self.optim_generator, self.optim_discriminators, self.scheduler_generator, self.scheduler_discriminator, self.dl, self.valid_dl = self.accelerator.prepare(
            self.msg_processor, self.detector, self.optim_generator, self.optim_discriminators, self.scheduler_generator, self.scheduler_discriminator, self.dl, self.valid_dl
        )

        self.discriminators = {k: self.accelerator.prepare(v) for k, v in self.discriminators.items()}

        hps = {"num_train_steps": num_train_steps, "num_warmup_steps": self.num_warmup_steps, "learning_rate": self.lr, "initial_learning_rate": self.initial_lr, "epochs": self.epochs}
        self.accelerator.init_trackers("SpeechTokenizer", config=hps)
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def device(self):
        return self.accelerator.device
    
    @property
    def nchunk_size(self):
        """Get nchunk_size from msg_processor, handling DDP wrapping"""
        return self.accelerator.unwrap_model(self.msg_processor).nchunk_size

    def warmup(self, step):
        if step < self.num_warmup_steps:
            return self.initial_lr + (self.lr - self.initial_lr) * step / self.num_warmup_steps
        else:
            return self.lr

    def log(self, values: dict, step, type=None, **kwargs):
        if not self.is_main:
            return  # Only main process has TensorBoard writer
        if type == 'figure':
            for k, v in values.items():
                self.writer.add_figure(k, v, global_step=step)
        elif type == 'audio':
            for k, v in values.items():
                self.writer.add_audio(k, v, global_step=step, **kwargs)
        else:
            for k, v in values.items():
                self.writer.add_scalar(k, v, global_step=step)

    def save(self, path, best_dev_loss, epoch):
        # Only main process saves checkpoints
        if not self.is_main:
            return
        if best_dev_loss < self.best_dev_loss:
            self.best_dev_loss = best_dev_loss
            # Save both msg_processor (embedder) and detector
            best_pkg = dict(
                msg_processor=self.accelerator.get_state_dict(self.msg_processor),
                detector=self.accelerator.get_state_dict(self.detector),
                epoch=epoch
            )
            torch.save(best_pkg, f'{self.results_folder}/embedder_best.pt')

        pkg = dict(
            embedder=self.accelerator.get_state_dict(self.msg_processor),
            detector=self.accelerator.get_state_dict(self.detector),  # Add watermark detector
            discriminators={k: self.accelerator.get_state_dict(v) for k, v in self.discriminators.items()},  # Rename for clarity
            optim_embedder=self.optim_generator.state_dict(),
            optim_detectors=self.optim_discriminators.state_dict(),
            scheduler_embedder=self.scheduler_generator.state_dict(),
            scheduler_detectors=self.scheduler_discriminator.state_dict(),
            best_dev_loss=self.best_dev_loss,
            epoch=epoch
        )
        torch.save(pkg, path)

    def validate(self):
        if self.is_main:
            print(f'\rValidate Epoch start...')
        self.generator.eval()
        self.msg_processor.eval()
        self.detector.eval()
        for d in self.discriminators.values():
            d.eval()

        total_mel_loss = 0.0
        total_cos_loss = 0.0
        total_adv_loss = 0.0
        total_dec_loss = 0.0
        total_vad_loss = 0.0
        total_vad_loss_neg = 0.0
        num = 0

        # Stats tracking for each attack
        # key: attack_name, value: {correct_chunks, total_chunks, pos_prob_sum, neg_prob_sum, neg_correct_chunks, neg_total_chunks, count}
        attack_stats = defaultdict(lambda: {
            "correct_chunks": 0, "total_chunks": 0, 
            "pos_prob_sum": 0.0,
            "neg_ori_prob_sum": 0.0, "neg_ori_count": 0,
            "neg_recon_prob_sum": 0.0, "neg_recon_count": 0,
            "neg_ori_correct_chunks": 0, "neg_ori_total_chunks": 0,
            "neg_recon_correct_chunks": 0, "neg_recon_total_chunks": 0,
            "loss": 0.0, "count": 0
        })
        total_pesq, total_stoi, total_snr = 0, 0, 0
        total_pesq_recon, total_stoi_recon, total_snr_recon = 0, 0, 0
        total_pesq_codec, total_stoi_codec, total_snr_codec = 0, 0, 0
        quality_count = 0

        with torch.inference_mode():
            # Read watermark ONCE for all validation batches to reduce I/O overhead
            with open("wmpool.txt", 'r') as wmp:
                watermark_data = eval(wmp.readline())
            # CRITICAL: Use int64 to match the dtype used during training forward pass
            watermark = torch.tensor(watermark_data, dtype=torch.int64).to(self.device)

            valid_steps = min(100, len(self.valid_dl))
            for i, batch in enumerate(tqdm(self.valid_dl, desc="Validation", total=valid_steps, ncols=100, disable=not self.is_main)):
                if i >= valid_steps: # Limit validation samples for full attack matrix to keep it fast
                    break
                ori_audio = torch.stack(batch).to(self.device)

                batch_size = ori_audio.size(0)
                message = watermark.unsqueeze(0).repeat(batch_size, 1)

                # ------------------- Generator forward -------------------
                reconstructed_audio, wm_audio, acoustic, acoustic_wm = self.generator(
                    ori_audio, message=message, msg_processor=self.msg_processor
                )
                min_len = min(ori_audio.shape[-1], wm_audio.shape[-1])
                ori_audio = ori_audio[..., :min_len]
                wm_audio = wm_audio[..., :min_len]
                recon_audio = reconstructed_audio[..., :min_len]

                # ------------------- Logging Audio and Spectrogram (Showpiece) -------------------
                if i == 0 and self.is_main:
                    for idx in range(min(batch_size, self.showpiece_num)):
                        # Log Audio
                        self.log({f"audio_recons/ori_{idx}": ori_audio[idx]}, step=self.steps.item(), type='audio', sample_rate=self.sample_rate)
                        self.log({f"audio_recons/wm_{idx}": wm_audio[idx]}, step=self.steps.item(), type='audio', sample_rate=self.sample_rate)
                        
                        # Log Spectrogram
                        ori_spec = mel_spectrogram(ori_audio[idx], **self.mel_kwargs)
                        wm_spec = mel_spectrogram(wm_audio[idx], **self.mel_kwargs)
                        
                        self.log({f"spec/ori_{idx}": plot_spectrogram(ori_spec.squeeze().cpu().numpy())}, step=self.steps.item(), type='figure')
                        self.log({f"spec/wm_{idx}": plot_spectrogram(wm_spec.squeeze().cpu().numpy())}, step=self.steps.item(), type='figure')

                # Calculate Quality Metrics (STOI and PESQ) on ALL validation samples
                if self.is_main:
                    for idx in range(batch_size):
                        # 1. Compare to Original (Ref Ori) - reflects overall fidelity
                        p_score, s_score, snr = compute_quality(ori_audio[idx], wm_audio[idx], self.sample_rate)
                        total_pesq += p_score
                        total_stoi += s_score
                        total_snr += snr
                        # 2. Compare to Reconstructed (Ref Recon) - reflects watermark quality purely
                        p_score_r, s_score_r, snr_r = compute_quality(recon_audio[idx], wm_audio[idx], self.sample_rate)
                        total_pesq_recon += p_score_r
                        total_stoi_recon += s_score_r
                        total_snr_recon += snr_r
                        # 3. Compare to Codec (Recon vs Ori) - reflects codec inherent loss
                        p_score_c, s_score_c, snr_c = compute_quality(ori_audio[idx], recon_audio[idx], self.sample_rate)
                        total_pesq_codec += p_score_c
                        total_stoi_codec += s_score_c
                        total_snr_codec += snr_c

                        quality_count += 1

                # Basic losses (Identity attack for base metrics)
                loss_mel = sum(map(lambda mel_k:mel_k[0] * mel_loss(ori_audio, wm_audio, **mel_k[1]), zip(self.multi_scale_mel_loss_lambdas, self.multi_scale_mel_loss_kwargs_list))) * self.mel_loss_lambda
                total_mel_loss += loss_mel.item()
                
                min_len_cos = min(acoustic_wm.shape[-1], acoustic.shape[-1])
                loss_cos = cos_loss(acoustic_wm[..., :min_len_cos], acoustic[..., :min_len_cos]) * self.cos_loss_lambda
                total_cos_loss += loss_cos.item()

                # Iterate through ALL attacks for comprehensive robustness check
                # target_chunks: [B, n_chunks]
                target_chunks = bits_to_chunks(message.float(), chunk_size=self.nchunk_size)
                
                # Comprehensive robustness check via ATTACK_REGISTRY
                # This loop now covers identity (CLEAN_AUDIO), vc_masking, and all other perturbations.
                for atk_name in ATTACK_REGISTRY.keys():
                    augmented_audio, vad_labels_atk, _ = attack_augmentation(
                        wm_audio, sample_rate=self.sample_rate, attack_name=atk_name, orig_audio=ori_audio
                    )
                    logits, chunk_logits = self.detect_watermark(augmented_audio, return_logits=True)
                    
                    # 1. Decoding performance
                    loss_dec_atk = decoding_loss(chunk_logits, target_chunks)
                    pred_chunks = chunk_logits.argmax(dim=-1)
                    correct_chunks = (pred_chunks == target_chunks).float().sum().item()
                    total_chunks = target_chunks.numel()
                    
                    # 2. Detection performance (Binary Classification)
                    # TP: Watermarked + Attack -> expect 1
                    detect_prob_pos = torch.sigmoid(logits).mean(dim=-1)
                    
                    # TN: Clean + Attack -> expect 0
                    is_ori = (i % 2 == 0)
                    neg_base = ori_audio if is_ori else reconstructed_audio
                    augmented_neg, _, _ = attack_augmentation(
                        neg_base, sample_rate=self.sample_rate, attack_name=atk_name
                    )
                    logits_neg, chunk_logits_neg = self.detect_watermark(augmented_neg, return_logits=True)
                    detect_prob_neg = torch.sigmoid(logits_neg).mean(dim=-1)
                    
                    # Also track "Bit Acc" for Clean audio as a sanity check (expect ~0.5)
                    pred_chunks_neg = chunk_logits_neg.argmax(dim=-1)
                    neg_correct = (pred_chunks_neg == target_chunks).float().sum().item()
                    neg_total = target_chunks.numel()

                    # Update stats
                    stats = attack_stats[atk_name]
                    stats["correct_chunks"] += correct_chunks
                    stats["total_chunks"] += total_chunks
                    stats["pos_prob_sum"] += detect_prob_pos.sum().item()
                    
                    if is_ori:
                        stats["neg_ori_prob_sum"] += detect_prob_neg.sum().item()
                        stats["neg_ori_count"] += batch_size
                        stats["neg_ori_correct_chunks"] += neg_correct
                        stats["neg_ori_total_chunks"] += neg_total
                    else:
                        stats["neg_recon_prob_sum"] += detect_prob_neg.sum().item()
                        stats["neg_recon_count"] += batch_size
                        stats["neg_recon_correct_chunks"] += neg_correct
                        stats["neg_recon_total_chunks"] += neg_total

                    stats["loss"] += loss_dec_atk.item()
                    stats["count"] += batch_size

                # Identity attack (baseline) is already tracked in the loop above.
                pass

        # ------------------- Print Detailed Diagnostics Table -------------------
        if self.is_main:
            print(f"\n{'='*115}")
            print(f"[VALIDATION DIAGNOSTICS - Step {int(self.steps.item())}]")
            print(f"{'Attack':<22} | {'Detect ACC':<12} | {'WM Bit Acc':<12} | {'WM Prob':<10} | {'Ori B.Acc':<10} | {'Ori Prob':<10} | {'Rec B.Acc':<10} | {'Rec Prob':<10}")
            print("-" * 135)
            
            # Categorized Sorting
            codec_keywords = ["encodec", "dac", "wavtokenizer"]
            codec_names = []
            other_names = []
            clean_name = "identity"

            for name in attack_stats.keys():
                if name == clean_name: continue
                if any(k in name.lower() for k in codec_keywords):
                    codec_names.append(name)
                else:
                    other_names.append(name)
            
            final_order = [clean_name] if clean_name in attack_stats else []
            final_order += sorted(other_names) + sorted(codec_names)

            for atk_name in final_order:
                stats = attack_stats[atk_name]
                count = stats["count"]
                if count == 0: continue
                
                wm_bit_acc = stats["correct_chunks"] / stats["total_chunks"] if stats["total_chunks"] > 0 else 0
                wm_prob = stats["pos_prob_sum"] / count
                
                ori_bit_acc = stats["neg_ori_correct_chunks"] / stats["neg_ori_total_chunks"] if stats["neg_ori_total_chunks"] > 0 else 0
                ori_prob = stats["neg_ori_prob_sum"] / stats["neg_ori_count"] if stats["neg_ori_count"] > 0 else 0
                
                recon_bit_acc = stats["neg_recon_correct_chunks"] / stats["neg_recon_total_chunks"] if stats["neg_recon_total_chunks"] > 0 else 0
                recon_prob = stats["neg_recon_prob_sum"] / stats["neg_recon_count"] if stats["neg_recon_count"] > 0 else 0
                
                # Detect Acc = (TP + TN) / 2
                tn_ori = 1.0 - ori_prob
                tn_recon = 1.0 - recon_prob
                det_acc = (wm_prob + (tn_ori + tn_recon)/2.0) / 2.0 if (stats["neg_ori_count"] > 0 and stats["neg_recon_count"] > 0) else wm_prob
                
                print(f"{atk_name:<22} | {det_acc:<12.4f} | {wm_bit_acc:<12.4f} | {wm_prob:<10.4f} | {ori_bit_acc:<10.4f} | {ori_prob:<10.4f} | {recon_bit_acc:<10.4f} | {recon_prob:<10.4f}")
            
            print(f"{'='*115}")
            
            log_dict = {} # Initialize log dictionary
            
            if quality_count > 0:
                print(f"  [Transparency - Ref Ori]")
                print(f"  - Avg PESQ (WB): {total_pesq / quality_count:.4f}")
                print(f"  - Avg STOI:      {total_stoi / quality_count:.4f}")
                print(f"  - Avg SI-SNR:    {total_snr / quality_count:.4f} dB")
                print(f"  [Transparency - Ref Recon]")
                print(f"  - Avg PESQ (WB): {total_pesq_recon / quality_count:.4f}")
                print(f"  - Avg STOI:      {total_stoi_recon / quality_count:.4f}")
                print(f"  - Avg SI-SNR:    {total_snr_recon / quality_count:.4f} dB")
                print(f"  [Codec Only - Recon vs Ori]")
                print(f"  - Avg PESQ (WB): {total_pesq_codec / quality_count:.4f}")
                print(f"  - Avg STOI:      {total_stoi_codec / quality_count:.4f}")
                print(f"  - Avg SI-SNR:    {total_snr_codec / quality_count:.4f} dB")
            # Add per-attack bit accuracy and loss to tensorboard/wandb
            for atk_name, stats in attack_stats.items():
                bit_acc = stats["correct_chunks"] / stats["total_chunks"] if stats["total_chunks"] > 0 else 0
                log_dict[f"val_acc/{atk_name}"] = bit_acc
                avg_l = stats["loss"] / stats["count"] if stats["count"] > 0 else 0
                log_dict[f"val_loss/{atk_name}"] = avg_l
            
            if quality_count > 0:
                log_dict["val_metrics/pesq"] = total_pesq / quality_count
                log_dict["val_metrics/stoi"] = total_stoi / quality_count
                log_dict["val_metrics/si_snr"] = total_snr / quality_count
                log_dict["val_metrics/pesq_recon"] = total_pesq_recon / quality_count
                log_dict["val_metrics/stoi_recon"] = total_stoi_recon / quality_count
                log_dict["val_metrics/si_snr_recon"] = total_snr_recon / quality_count
                log_dict["val_metrics/pesq_codec"] = total_pesq_codec / quality_count
                log_dict["val_metrics/stoi_codec"] = total_stoi_codec / quality_count
                log_dict["val_metrics/si_snr_codec"] = total_snr_codec / quality_count
            
            # Log aggregate base losses (Identity)
            log_dict["val_metrics/mel_loss"] = total_mel_loss / valid_steps
            log_dict["val_metrics/cos_loss"] = total_cos_loss / valid_steps
                
            self.log(log_dict, step=self.steps.item())

        identity_loss = attack_stats["identity"]["loss"] / attack_stats["identity"]["count"] if attack_stats["identity"]["count"] > 0 else 0
        return identity_loss

    def train(self):
        # print(torch.cuda.is_available())  # True
        # print(torch.cuda.current_device())  # 0
        # print(torch.cuda.get_device_name(0))
        self.generator.train()  # RNN backward requires train mode even if frozen
        # If there are any normalization layers, we should keep them in eval mode for stability
        for m in self.generator.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                m.eval()
        for d in self.discriminators.values():
            d.train()

        # Scheduler and warmup logic is handled inside the training loop after forward/backward
        steps = int(self.steps.item())
        if steps < self.num_warmup_steps:
            lr = self.warmup(steps)
            for pg in self.optim_generator.param_groups: pg['lr'] = lr
            for pg in self.optim_discriminators.param_groups: pg['lr'] = lr
        else:
            lr = self.scheduler_generator.get_last_lr()[0]
        for epoch in range(self.epochs):
            # Per-attack accuracy tracker: {attack_name: {"correct": int, "total": int}}
            attack_acc_tracker = defaultdict(lambda: {"correct": 0, "total": 0})
            # waveform = [batch, 1, T]
            for i, audio in enumerate(tqdm(self.dl, desc=f"Epoch {epoch}", ncols=100, disable=not self.is_main)):
                # ------------------- Prepare batch -------------------
                ori_audio = torch.stack(audio).to(self.device)

                # random 16-bit watermark/message
                batch_size = ori_audio.size(0)
                n_bits = 16  # 16 bits message
                message = torch.randint(0, 2, (batch_size, n_bits), dtype=torch.int64).to(self.device)

                # ------------------- Freeze the generator (already frozen in __init__) -------------------
                # Generator is not DDP-wrapped, just a regular model on device

                # ------------------- Embedder forward -------------------
                self.optim_generator.zero_grad()
                reconstructed_audio, wm_audio, acoustic, acoustic_wm = self.generator(
                    ori_audio, message=message, msg_processor=self.msg_processor
                )

                min_len = min(ori_audio.shape[-1], wm_audio.shape[-1], reconstructed_audio.shape[-1])
                ori_audio = ori_audio[..., :min_len]
                wm_audio = wm_audio[..., :min_len]
                reconstructed_audio = reconstructed_audio[..., :min_len]

                # ------------------- Compute Generator Adversarial Loss FIRST (before D update) -------------------
                # Freeze discriminator params to avoid gradient leakage
                for d in self.discriminators.values():
                    for p in d.parameters():
                        p.requires_grad = False

                loss_adv = 0.0
                loss_fm = 0.0
                for d in self.discriminators.values():
                    # d(y, y_hat) -> (real_preds_list, fake_preds_list, real_fmaps, fake_fmaps)
                    _, fake_preds_list, real_fmaps, fake_fmaps = d(reconstructed_audio, wm_audio)  # No detach — need gradients to flow to generator
                    for fake_pred in fake_preds_list:
                        loss_adv += adversarial_loss_g(fake_pred)
                    # Feature matching loss for training stability
                    loss_fm += feature_loss(real_fmaps, fake_fmaps)
                loss_adv = (loss_adv + loss_fm * 2.0) * self.adv_loss_lambda

                # ------------------- Train mel loss -------------------
                loss_mel = sum(map(lambda mel_k:mel_k[0] * mel_loss(reconstructed_audio, wm_audio, **mel_k[1]), zip(self.multi_scale_mel_loss_lambdas, self.multi_scale_mel_loss_kwargs_list))) * self.mel_loss_lambda

                # ------------------ Train cos loss ---------------------
                min_len = min(acoustic_wm.shape[-1], acoustic.shape[-1])
                acoustic_wm_aligned = acoustic_wm[..., :min_len]
                acoustic_aligned = acoustic[..., :min_len]
                loss_cos = cos_loss(acoustic_wm_aligned, acoustic_aligned) * self.cos_loss_lambda

                # ---------------- Train Decoding and VAD loss -----------------
                # 1. Positive Sample Augmentation (Watermarked Audio)
                augmented_audio, vad_labels, atk_name = attack_augmentation(
                    wm_audio, sample_rate=self.sample_rate, attack_name=None, orig_audio=ori_audio
                )
                logits, chunk_logits = self.detect_watermark(augmented_audio, return_logits=True)
                
                message_float = message.float()
                target_chunks = bits_to_chunks(message_float, chunk_size=self.nchunk_size)
                loss_dec = decoding_loss(chunk_logits, target_chunks) * self.dec_loss_lambda
                
                # Independent alignment for positive sample
                min_lens_pos = min(logits.shape[-1], vad_labels.shape[-1])
                loss_vad_pos = vad_based_loss(logits[..., :min_lens_pos], vad_labels[..., :min_lens_pos], from_logits=True) * self.vad_loss_lambda

                # 2. ENHANCED NEGATIVE SUPERVISION (Alternating to save memory)
                # Now covers both VCTK and LibriTTS (if using mixed dataset)
                if steps % 2 == 0:
                    augmented_neg, _, _ = attack_augmentation(ori_audio, sample_rate=self.sample_rate, attack_name=None)
                else:
                    augmented_neg, _, _ = attack_augmentation(reconstructed_audio, sample_rate=self.sample_rate, attack_name=None)
                
                neg_logits, _ = self.detect_watermark(augmented_neg, return_logits=True)
                # Independent alignment for negative sample
                vad_labels_neg = torch.zeros_like(neg_logits) # Ensure labels match actual output length
                min_lens_neg = min(neg_logits.shape[-1], vad_labels_neg.shape[-1])
                loss_vad_neg = vad_based_loss(neg_logits[..., :min_lens_neg], vad_labels_neg[..., :min_lens_neg], from_logits=True) * self.vad_loss_lambda

                # ---------------- accumulate total loss and backward GENERATOR ---------------------
                total_loss = loss_mel + loss_cos + loss_adv + loss_dec + loss_vad_pos + loss_vad_neg

                # Use accelerator backward — must happen BEFORE D update to avoid in-place modification error
                # Use no_sync on discriminators to skip unnecessary AllReduce (~11M params)
                with ExitStack() as stack:
                    for d in self.discriminators.values():
                        stack.enter_context(self.accelerator.no_sync(d))
                    self.accelerator.backward(total_loss)
                
                # NOW Unfreeze discriminator params for the subsequent Discriminator update step
                for d in self.discriminators.values():
                    for p in d.parameters():
                        p.requires_grad = True

                # Gradient norm logging (only every 1000 steps to reduce overhead)
                if self.is_main and steps % 1000 == 0:
                    total_norm = 0
                    for p in self.msg_processor.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    self.log({"train/embedder_grad_norm": total_norm}, step=steps)

                    detector_grad_norm = 0
                    for p in self.detector.parameters():
                        if p.grad is not None:
                            detector_grad_norm += p.grad.data.norm(2).item() ** 2
                    detector_grad_norm = detector_grad_norm ** 0.5
                    self.log({"train/detector_grad_norm": detector_grad_norm}, step=steps)

                self.optim_generator.step()

                # ------------------- Train Discriminators (AFTER generator backward) -------------------
                self.optim_discriminators.zero_grad()
                loss_D = 0.0
                # Use no_sync on G models to skip unnecessary AllReduce
                with self.accelerator.no_sync(self.msg_processor), self.accelerator.no_sync(self.detector):
                    for d in self.discriminators.values():
                        real_preds_list, fake_preds_list, _, _ = d(reconstructed_audio, wm_audio.detach())
                        for real_pred, fake_pred in zip(real_preds_list, fake_preds_list):
                            loss_D += adversarial_loss_d(real_pred, fake_pred)
                    self.accelerator.backward(loss_D)

                self.optim_discriminators.step()

                # ------------------- LR logging -------------------
                for param_group in self.optim_generator.param_groups:
                    lr_g = param_group["lr"]
                for param_group in self.optim_discriminators.param_groups:
                    lr_d = param_group["lr"]

                # ------------------- Logging (log() already guards is_main) -------------------
                self.log({"lr/generator": lr_g, "lr/discriminator": lr_d}, step=steps)
                self.log({
                    "train/vad_loss": loss_vad_pos.item(),
                    "train/vad_loss_neg": loss_vad_neg.item(),
                    "train/cos_loss": loss_cos.item(),
                    "train/mel_loss": loss_mel.item(),
                    "train/decoding_loss": loss_dec.item(),
                    "train/d_loss": loss_D.item(),
                    "train/adv_loss": loss_adv.item(),
                    "train/feature_matching_loss": loss_fm,
                    "train/total_loss": total_loss.item()
                }, step=steps)


                
                # ------------------- Trigger detailed validation every 1000 steps -------------------
                if steps % 1000 == 0 and steps != 0:
                    self.accelerator.wait_for_everyone()
                    self.generator.eval()
                    self.msg_processor.eval()
                    self.detector.eval()
                    for d in self.discriminators.values():
                        d.eval()

                    val_loss_sum = self.validate()
                    
                    self.save(
                        self.results_folder / f'WatermarkTrainer_{steps:08d}.pt',
                        val_loss_sum,
                        epoch
                    )

                    # Switch back to train mode
                    self.generator.train()
                    # Re-apply the stability strategy for frozen generator: 
                    # keep normalization layers in eval mode.
                    for m in self.generator.modules():
                        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                            m.eval()

                    self.msg_processor.train()
                    self.detector.train()
                    for d in self.discriminators.values():
                        d.train()
                    
                    self.accelerator.wait_for_everyone()

                # Update lr    
                self.steps += 1
                steps = int(self.steps.item())

                # Hard termination condition: stop training at 150,000 steps
                if steps >= 150000:
                    self.accelerator.wait_for_everyone()
                    if self.is_main:
                        print(f"\n[Terminating] Reached target steps: {steps}. Saving and exiting...")
                        val_loss_sum = self.validate()
                        self.save(
                            self.results_folder / f'WatermarkTrainer_final_{steps:08d}.pt',
                            val_loss_sum,
                            epoch
                        )
                    self.accelerator.wait_for_everyone()
                    return # Exit training method
                
                if steps < self.num_warmup_steps:
                    lr = self.warmup(steps)
                    for param_group in self.optim_generator.param_groups:
                        param_group['lr'] = lr
                    for param_group in self.optim_discriminators.param_groups:
                        param_group['lr'] = lr
                else:
                    self.scheduler_discriminator.step() 
                    self.scheduler_generator.step() 
                    lr = self.scheduler_generator.get_last_lr()[0]  

    def continue_train(self):
        self.load()
        self.train()

    def detect_watermark(
        self, x: torch.Tensor, return_logits=False
    ):
        # Unwrap DDP-wrapped models to access custom methods
        generator = self.accelerator.unwrap_model(self.generator)
        embedding = generator.forward_feature(x)
        
        if return_logits:
            # CRITICAL FIX: Use the wrapped detector (self.detector) instead of the unwrapped one
            # to ensure DDP gradient synchronization during training and validation.
            return self.detector(embedding)
        
        # When return_logits=False (inference/testing), we need the convenience method
        # which is only available on the unwrapped model.
        detector = self.accelerator.unwrap_model(self.detector)
        return detector.detect_watermark(embedding)
