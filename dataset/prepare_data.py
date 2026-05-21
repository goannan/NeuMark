import argparse
from pathlib import Path

import torchaudio


DEFAULT_LIBRITTS_TRAIN_SUBSETS = ["train-clean-100", "train-clean-360", "train-other-500"]
DEFAULT_LIBRISPEECH_TEST_SUBSETS = ["test-clean"]
DEFAULT_TRAIN_VALID_SIZE = 100


def download_libritts(root: Path, subsets: list[str]) -> None:
    for subset in subsets:
        print(f"[dataset] downloading/checking LibriTTS {subset} ...")
        torchaudio.datasets.LIBRITTS(root=str(root), url=subset, download=True)


def download_librispeech(root: Path, subsets: list[str]) -> None:
    for subset in subsets:
        print(f"[dataset] downloading/checking LibriSpeech {subset} ...")
        torchaudio.datasets.LIBRISPEECH(root=str(root), url=subset, download=True)


def collect_audio_files(root: Path, dataset_name: str, subsets: list[str]) -> list[Path]:
    files: list[Path] = []
    for subset in subsets:
        subset_dir = root / dataset_name / subset
        files.extend(subset_dir.rglob("*.flac"))
        files.extend(subset_dir.rglob("*.wav"))
    return sorted(files)


def write_filelist(path: Path, files: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for audio_path in files:
            f.write(f"{audio_path.resolve()}\n")
    print(f"[dataset] wrote {len(files)} files -> {path}")


def parse_subsets(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NeuMark datasets and create file lists.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--libritts-train-subsets", type=str, default=",".join(DEFAULT_LIBRITTS_TRAIN_SUBSETS))
    parser.add_argument("--librispeech-test-subsets", type=str, default=",".join(DEFAULT_LIBRISPEECH_TEST_SUBSETS))
    parser.add_argument("--train-valid-size", type=int, default=DEFAULT_TRAIN_VALID_SIZE)
    parser.add_argument("--filelist-dir", type=Path, default=Path(__file__).resolve().parent / "filelists")
    args = parser.parse_args()

    root = args.root.resolve()
    libritts_train_subsets = parse_subsets(args.libritts_train_subsets)
    librispeech_test_subsets = parse_subsets(args.librispeech_test_subsets)

    root.mkdir(parents=True, exist_ok=True)
    download_libritts(root, libritts_train_subsets)
    download_librispeech(root, librispeech_test_subsets)

    all_train_files = collect_audio_files(root, "LibriTTS", libritts_train_subsets)
    valid_files = all_train_files[:args.train_valid_size]
    train_files = all_train_files[args.train_valid_size:]
    test_files = collect_audio_files(root, "LibriSpeech", librispeech_test_subsets)

    if len(all_train_files) <= args.train_valid_size:
        raise RuntimeError(f"No training audio found under {root}")
    if not valid_files:
        raise RuntimeError(f"No train-validation audio found under {root}")
    if not test_files:
        raise RuntimeError(f"No LibriSpeech test audio found under {root}")

    write_filelist(args.filelist_dir / "train.txt", train_files)
    write_filelist(args.filelist_dir / "valid.txt", valid_files)
    write_filelist(args.filelist_dir / "test.txt", test_files)

    print("[dataset] done")


if __name__ == "__main__":
    main()
