import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from STmodels.model import SpeechTokenizer
from STmodels.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator, MultiScaleSTFTDiscriminator
import json
import argparse
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from train import WMTrainer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default=str(PROJECT_ROOT / "config/default.json"), help='Config file path')
    parser.add_argument('--audio_dir', type=str, default=str(PROJECT_ROOT / "dataset/LibriTTS"), help='Audio folder path')
    parser.add_argument('--exts', type=str, help="Audio file extensions, splitting with ','", default='wav')
    parser.add_argument('--valid_set_size', type=float, default=1000)
    parser.add_argument('--continue_train', action='store_true', help='Continue to train from checkpoints')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    cfg["train_files"] = str(resolve_project_path(cfg.get("train_files")))
    cfg["valid_files"] = str(resolve_project_path(cfg.get("valid_files")))
    cfg["results_folder"] = str(resolve_project_path(cfg.get("results_folder")))
    exts = args.exts.split(',')
    path = Path(args.audio_dir)
    file_list = [str(file) for ext in exts for file in path.glob(f'**/*.{ext}')]
    if args.valid_set_size != 0 and args.valid_set_size < 1:
        valid_set_size = int(len(file_list) * args.valid_set_size)
    else:
        valid_set_size = int(args.valid_set_size)
    train_file_list = cfg.get('train_files')
    valid_file_list = cfg.get('valid_files')
    sample_rate = cfg.get('sample_rate')
    segment_size = cfg.get('segment_size')
    if not (os.path.exists(train_file_list) and os.path.exists(valid_file_list)):
        if not file_list:
            raise FileNotFoundError(
                f"No audio files found in {path}. Run bash dataset/prepare_data.sh first, "
                "or pass --audio_dir to an existing audio folder."
            )
        os.makedirs(os.path.dirname(train_file_list), exist_ok=True)
        os.makedirs(os.path.dirname(valid_file_list), exist_ok=True)
        try:
            for i, audio_file in tqdm(enumerate(file_list)):

                if i < valid_set_size:
                    with open(valid_file_list, 'a+') as f:
                        f.write(f'{audio_file}\n')
                else:
                    with open(train_file_list, 'a+') as f:
                        f.write(f'{audio_file}\n')
        except Exception as e:
            print(f"Error in processing {audio_file}, error message: {e}")
    config_path = PROJECT_ROOT / "STmodels/pretrained_model/speechtokenizer_hubert_avg_config.json"
    ckpt_path = PROJECT_ROOT / "STmodels/pretrained_model/SpeechTokenizer.pt"
    generator = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)

    discriminators = {'mpd': MultiPeriodDiscriminator(), 'msd': MultiScaleDiscriminator(),
                      'mstftd': MultiScaleSTFTDiscriminator(32)}

    trainer = WMTrainer(generator=generator,
                        discriminators=discriminators,
                        cfg=cfg)

    if args.continue_train:
        trainer.continue_train()
    else:
        trainer.train()
