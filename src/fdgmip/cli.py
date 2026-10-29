import argparse

from fdgmip.config import Config, load_config
from fdgmip.logging_utils import setup_logging
from fdgmip.provenance import seed_everything


def parse(description: str, argv=None, add=None) -> tuple[argparse.Namespace, Config]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("-c", "--config", action="append", required=True,
                    help="YAML config; repeatable, later files override earlier ones")
    ap.add_argument("--log-level", default="INFO")
    if add:
        add(ap)
    args = ap.parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(*args.config)
    seed_everything(int(cfg.get("seed")))
    return args, cfg
