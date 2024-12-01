import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import numpy as np
from joblib import dump, load
from glob import glob
import json
import os
import torch
from logger import TensorLogger
import torch.optim as optim
from config import Config
from model_token_ids import TweetsBlock
from sklearn.metrics import matthews_corrcoef, accuracy_score

# Model hyperparameters
class HyperParams:
    def __init__(self):
        self.T=30
        self.input_dim=6
        self.hidden_dim=128
        self.num_layers=4
        self.output_dim=2
        self.batch_size = 32
        self.T = 30
        self.dropout = 0.3
        self.lr = 0.01

class SimpleNN(nn.Module):
    def __init__(self, cfg: Config, params: HyperParams):
        super(SimpleNN, self).__init__()
        self.hyperparams = params
        
        # LSTM layer
        self.lstm = nn.LSTM(self.hyperparams.input_dim, self.hyperparams.hidden_dim, self.hyperparams.num_layers, batch_first=True, device="cpu")
        self.lstm2 = nn.LSTM(5, 32, 2, batch_first=True, device="cpu")
        self.tweets_block = TweetsBlock()
        
        # Fully connected layer to downscale to output dimension
        self.fc1 = nn.Linear(160, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
    
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(self.hyperparams.dropout)
        
        self._initialize_weights()
        self._setup_general(cfg.ckpt_path, cfg.log_dir, lr=self.hyperparams.lr)

    def _initialize_weights(self):
        # Kaiming initialization often works better with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, X_nums, X_news_nums, X_news_tkids):
        # Main path
        _, (h_n, _) = self.lstm(X_nums)
        _, (h_n2, _) = self.lstm2(X_news_nums)
        # X_cnn_out = self.tweets_block(X_news_tkids) 
        
        x = torch.cat([h_n[-1], h_n2[-1]], dim=1)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x
    
    def comp_loss(self, outputs, targets):
        """
        Compute a combined loss that penalizes both regression error 
        and classification accuracy
        
        Args:
        - outputs: Model predictions (continuous values)
        - targets: Ground truth values
        """
        # Regression loss (Mean Squared Error)
        regression_loss = F.mse_loss(outputs, targets)
        
        # Binary Classification Loss
        # Create binary labels based on threshold
        binary_targets = (targets > self.threshold).float()
        binary_outputs = (outputs.reshape(-1) > self.threshold).float()
        
    #    Compute classification loss (Binary Cross Entropy)
        classification_loss = F.binary_cross_entropy(
            binary_outputs, 
            binary_targets
        )
        
        # Combine losses
        total_loss = (
            self.regression_weight * 
            regression_loss 
            + 
            self.classification_weight * classification_loss
        )
        
        return total_loss
    
    def _setup_general(self, ckp_dir, log_dir, lr):
        self.threshold = 0.5194457065059035
        self.regression_weight = 0.95
        self.classification_weight = 1.0
        # Setup checkpointing
        self.logger = TensorLogger(logdir=log_dir)
        self.checkpoint_dir = ckp_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)
        
    def setup_criterion(self, class_weights = None):
        self.criterion = nn.MSELoss()
    
    def setup4newfit(self):
        self.best_loss = float(np.inf)
        self.best_model_state = None
        self.n_trained_epochs = 0
        self.global_step = 1
        
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr  # Set your desired learning rate here
        
    def fit(self, train_loader, val_loader, test_loader, n_epochs=100, lr=None):
        # Reduce LR when validation loss plateaus
        if lr is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6,
            )
        if lr:
            self.set_lr(lr)

        for epoch in range(n_epochs + self.n_trained_epochs)[self.n_trained_epochs:]:
            self.train()
            train_loss = 0.0

            for X_nums, X_news_nums, X_news_tkids, Y in train_loader:
                self.optimizer.zero_grad()
                outputs = self(X_nums, X_news_nums, X_news_tkids)
                Y = Y.to("cpu")
                loss = self.comp_loss(outputs, Y.reshape(-1))
                loss.backward()
                self.logger.log_ud(self, self.global_step, self.lr)
                self.optimizer.step()
                self.global_step += 1
                train_loss += loss.item() * Y.size(0)

            train_loss = train_loss / len(train_loader.dataset)

            # Validation loop
            self.eval()
            val_loss = 0.0
            Y_truth = []
            Y_pred = []
            with torch.no_grad():
                for X_nums, X_news_nums, X_news_tkids, Y in val_loader:
                    outputs = self(X_nums, X_news_nums, X_news_tkids)
                    loss = self.comp_loss(outputs, Y.reshape(-1))
                    val_loss += loss.item() * Y.size(0)
                    Y_truth.extend(self.threshold < Y)
                    Y_pred.extend(self.threshold < outputs.reshape(-1))

            val_loss = val_loss / len(val_loader.dataset)
            if lr is None:
                scheduler.step(val_loss)
            val_acc = accuracy_score(Y_truth, Y_pred)
            val_mcc = matthews_corrcoef(Y_truth, Y_pred)

            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val acc: {val_acc:.4f}, Val mcc: {val_mcc:.4f}')
            self.logger.log_metric(train_loss, self.global_step)
            self.logger.log_metric(val_loss, self.global_step, mode="Val")
            self.logger.log_metric(val_acc, self.global_step, metric="acc", mode="Val")
            self.logger.log_metric(val_mcc, self.global_step, metric="mcc", mode="Val")

            # Save the best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                # Save only the model's state dict
                self.best_model_state = self.state_dict().copy()
                checkpoint_path = self._save_checkpoint(epoch, val_loss, val_acc)
                print(f"New best model saved: {checkpoint_path}")


    def test(self, test_loader, load_best=True):
         # Load the best model state that was saved during training
        if load_best:
        # Load the best model state
            self.initialize_from_ckp()
            
        # Test loop
        self.eval()
        test_loss = 0.0
        Y_truth = []
        Y_pred = []
        with torch.no_grad():
            for X_nums, X_news_nums, X_news_tkids, Y in test_loader:
                outputs = self(X_nums, X_news_nums, X_news_tkids)
                loss = self.comp_loss(outputs, Y.reshape(-1))
                test_loss += loss.item() * Y.size(0)
                Y_truth.extend(self.threshold < Y)
                Y_pred.extend(self.threshold < outputs.reshape(-1))

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = accuracy_score(Y_truth, Y_pred)
        test_mcc = matthews_corrcoef(Y_truth, Y_pred)

        print(f'Loss: {test_loss:.4f}, acc: {test_acc:.4f}, mcc: {test_mcc:.4f}')
        self.logger.log_metric(test_loss, self.global_step, mode="Test")
        self.logger.log_metric(test_acc, self.global_step, metric="acc", mode="Test")
        self.logger.log_metric(test_mcc, self.global_step, metric="mcc", mode="Test")


    def initialize_from_ckp(self):
        """Initialize a new model or load the latest checkpoint if available"""
        checkpoint_info = self._find_latest_checkpoint()
        
        if checkpoint_info:
            print(f"Resuming from checkpoint: {checkpoint_info['checkpoint_path']}")
            self._load_checkpoint(checkpoint_info['checkpoint_path'])
        else:
            self.setup4newfit()
    
    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the checkpoint directory"""
        # Look for checkpoint files
        checkpoint_files = glob(os.path.join(self.checkpoint_dir, 'model_*.pth'))
        
        if not checkpoint_files:
            return None
            
        # Get the latest checkpoint based on modification time
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        return {
            'checkpoint_path': latest_checkpoint
        }
    
    def _save_checkpoint(self, epoch, loss, acc):
        """Save model checkpoint and training state"""
        # Save model
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'model_epoch_{epoch}_loss_{loss:.4f}_acc_{acc:.4f}.pth'
        )
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'global_step': self.global_step,
        }
        torch.save(checkpoint, checkpoint_path)
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model and training state from checkpoint"""
        # Loading full checkpoint
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.n_trained_epochs = checkpoint['epoch']
        self.best_loss = checkpoint['loss']
        self.global_step = checkpoint['global_step']
        self.best_model_state = self.state_dict()
    
    def _cleanup_old_checkpoints(self, keep_n=5):
        """Keep only the N most recent checkpoints"""
        checkpoint_files = glob(os.path.join(self.checkpoint_dir, 'model_*.pth'))
        if len(checkpoint_files) <= keep_n:
            return
            
        # Sort checkpoints by modification time
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Remove older checkpoints
        for checkpoint_file in checkpoint_files[keep_n:]:
            os.remove(checkpoint_file)
    
