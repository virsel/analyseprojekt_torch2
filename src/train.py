import torch
import os
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import __init__
from data import get_dataloaders, get_data
import config as m_config
from model import get_model
from trainer import get_trainer
from logger import Logger

import logging
logging.basicConfig(level=logging.DEBUG)
version = 1
    
if __name__ == '__main__':
    # Set environment variables for distributed training
    os.environ["MASTER_PORT"] = "29500"  # Default port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["WORLD_SIZE"] = "4"
    os.environ["NODE_RANK"] = "0"
    # laden der Konfiguration
    config = m_config.get_default_config()

    X_train, y_train, X_val, y_val = get_data(config)
    
    train_loader, val_loader, class_weights = get_dataloaders(X_train, y_train, X_val, y_val)

    config.modelConfig.set(X_train.shape[1], len(np.unique(y_train)), class_weights)
    
    model = get_model(config.modelConfig, val_loader)
    logger = Logger(model, config.log_dir)

    # train with pytorch lightning
    trainer = get_trainer(config, logger=logger)
    trainer.fit(model, train_loader, val_loader, ckpt_path=config.ckpt_path)