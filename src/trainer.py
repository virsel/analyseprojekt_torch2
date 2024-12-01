from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from config import Config

def get_ckpt_cb(dir):
    return ModelCheckpoint(
        monitor="val_loss",
        dirpath=dir,
        filename='ckpt-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode="min",
    )

def get_trainer(cfg: Config, logger=None, strategy="ddp_spawn"):
    checkpoint_callback = get_ckpt_cb(cfg.ckpt_path)
    return Trainer(
        # checkpointing
        callbacks=[checkpoint_callback],
        accelerator='cpu',
        # distribution
        strategy=strategy,
        devices=cfg.n_workers,
        # um_sanity_val_steps=0,  # Disable sanity check temporarily
        # num_workers=0,           # Disable multi-processing
        # enable_progress_bar=True, # Enable progress bar
        max_epochs=10,
        # logging
        logger=logger
    )
    
def get_test_trainer(cfg: Config, logger=None):
    return Trainer(
        accelerator='cpu',
        strategy='auto',
        logger=logger
    )
    