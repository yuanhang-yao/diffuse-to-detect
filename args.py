import argparse
import os
import time


DEFAULT_DATASET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Ours", "dataset")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alclnet", choices=["alclnet", "dnanet"])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--test_start_epoch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    args = parser.parse_args()
    print("=" * 48)
    print(f"Time : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PID  : {os.getpid()}")
    print("-" * 48)
    print("Arguments:")
    print("-" * 48)
    for key, value in vars(args).items():
        print(f"{key:<22}: {value}")
    print("=" * 48)
    return args
