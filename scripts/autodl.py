"""Small Python entry point for AutoDL. Preflight is the default, training is explicit."""
import argparse
import json

from routemind.training import config_from, preflight, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "train"], default="check", nargs="?")
    parser.add_argument("--config", default="configs/qwen3_32b.json")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = config_from(args.config)
    if args.action == "check":
        print(json.dumps(preflight(config, args.allow_unreviewed), indent=2))
        try:
            import torch
            print(json.dumps({"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
                              "gpu_count": torch.cuda.device_count()}))
            if torch.cuda.is_available():
                for index in range(torch.cuda.device_count()):
                    prop = torch.cuda.get_device_properties(index)
                    print(f"GPU {index}: {prop.name}; VRAM {prop.total_memory / 2**30:.1f} GiB")
        except ImportError:
            print("PyTorch is not installed; CPU data scripts remain usable.")
    else:
        run_training(config, args.allow_unreviewed, args.resume)


if __name__ == "__main__":
    main()
