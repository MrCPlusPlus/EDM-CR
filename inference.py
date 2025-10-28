import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import EDMCRSampler

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--bs", type=int, default=2, help="Batch size.")
    parser.add_argument(
            "--task",
            type=str,
            default="cloud_removal",
            choices=['cloud_removal'],
            help="task name.",
            )
    parser.add_argument(
            "--ckpt_path",
            type=str,
            default="./logs",
            help="Path for loading checkpoint.",
            )
    parser.add_argument(
            "--config_path",
            type=str,
            default="",
            help="Path for config file.",
            )
    args = parser.parse_args()

    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()

    if args.task == 'cloud_removal':
        configs = OmegaConf.load(args.config_path)
        ckpt_path = args.ckpt_path
    else:
        raise TypeError(f"Unexpected task type: {args.task}!")

    configs.model.ckpt_path = str(ckpt_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    edmcr_sampler = EDMCRSampler(
            configs,
            use_amp=True,
            seed=args.seed,
            padding_offset=configs.model.params.get('cloudy_size', 256),
            )

    edmcr_sampler.inference(
            args.in_path,
            args.out_path,
            bs=args.bs,
            noise_repeat=False
            )

if __name__ == '__main__':
    main()
