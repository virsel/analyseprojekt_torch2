import torch.nn as nn
import torch
from transformers import AutoModel

import os
import pickle
import ast

random_state=42
dir_path = os.path.dirname(os.path.abspath(__file__))
emb_path = os.path.join(dir_path, '../input/embtable.pt')
mapping_path = os.path.join(dir_path, '../input/embtable_mapping.pkl')

# Loading the precomputed embeddings during model initialization
def load_token_embeddings():
    return torch.load(emb_path)

def load_mapping():
    # Load from a pickle file
    with open(mapping_path, 'rb') as file:
        res = pickle.load(file)
    return res

    
    
class TweetsBlock(nn.Module):
    def __init__(self, 
                 context_length=40,
                 dropout_rate=0.3):
        super(TweetsBlock, self).__init__()
        

        # Load embeddings
        token_embeddings = load_token_embeddings()
        
        # Create embedding layer
        self.embedding = nn.Embedding.from_pretrained(token_embeddings, freeze=True)
        
        self.tk_mapping = load_mapping()

        
        # CNN layer for feature extraction
        self.cnn = nn.Conv1d(
            in_channels=768, 
            out_channels=16, 
            kernel_size=3, 
            padding=1
        )
        
        
        # Gradual dimension reduction
        self.fc1 = nn.Linear(16 * context_length, 256)
        self.fc2 = nn.Linear(256, 128)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Activation functions
        self.relu = nn.ReLU()
        
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization often works better with ReLU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, X_tkids):
        # Map real token IDs to embedding table IDs
        # 3 is <unk> token

        res_messages = torch.tensor([])
        for j in range(X_tkids.shape[1]):
            X_tkids_message = X_tkids[:, j, :]
            mapped_tkids = torch.tensor([
                 [self.tk_mapping.get(int(tkid), 3) for tkid in batch]
                 for batch in X_tkids_message.cpu().numpy()
             ]).to(X_tkids.device)

            # Get embeddings using the mapped IDs
            token_embeddings = self.embedding(mapped_tkids)

            # Transpose for CNN (batch, hidden_size, seq_len)
            token_embeddings = token_embeddings.transpose(1, 2)

            # Apply CNN
            cnn_features = self.cnn(token_embeddings)

            # Flatten
            x_oneday = cnn_features.view(cnn_features.size(0), -1)
            
            res_messages = torch.cat((res_messages, x_oneday.unsqueeze(1)), 1)
                
            # compute avg of all messages
        res = torch.mean(res_messages, 1)
          
          
        x = self.fc1(res)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)



        return x