from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Config:
    version: str
    output_folder: str
    log_dir: str
    ckpt_path: str


def get_default_config(root="./", version="v3") -> Config:
    output_folder = Path(f"{root}../output")

    cfg = Config(
        version=version,
        # output
        output_folder=str(output_folder),
        ckpt_path=str(output_folder / version / "checkpoints"),
        log_dir=str(output_folder / version / "logs"),
    )

    # Ensure the checkpoint directory exists, create it if not
    os.makedirs(cfg.ckpt_path, mode=0o777, exist_ok=True)

    return cfg
