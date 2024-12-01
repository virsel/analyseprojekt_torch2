from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')
    
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import ast  # for safely evaluating string representations of lists
from sklearn.model_selection import train_test_split


random_state=42
dir_path = os.path.dirname(os.path.abspath(__file__))
data_stock_path = os.path.join(dir_path, '../input/stock_step1.csv')
data_news_path = os.path.join(dir_path, '../input/news2_step2b.csv')



def load_data(batch_size=64):
    # Read data
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'adjusted']
    df_stock = pd.read_csv(data_stock_path, usecols=['date', 'target', *feature_columns])
    df_news = pd.read_csv(data_news_path)
    df_news.loc[:, 'news_tkids'] = df_news['news_tkids'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.int64))
    # Convert date strings to datetime objects before comparison
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_news['date'] = pd.to_datetime(df_news['date'])
    
    # Scale the numeric columns
    df_stock[feature_columns] = MinMaxScaler().fit_transform(df_stock[feature_columns])
    scaler = MinMaxScaler().fit(df_stock[['target']])
    df_stock[['target']] =scaler.transform(df_stock[['target']])
    rise_threshold = scaler.transform(np.array([0]).reshape(-1,1))[0][0]
    print(f"Rise threshold: {rise_threshold}")

    # Create training data for this stock
    data = create_training_data(df_stock, df_news)
    
    # Split data into train+validation and test sets (80-20 split)
    train_val_size = int(len(data) * 0.8)
    train_val_data = data[:train_val_size]
    test_data = data[train_val_size:]
    
    # Further split train+validation into train and validation (85-15 split)
    train_size = int(len(train_val_data) * 0.85)
    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]
    
    # Prepare the dataloaders
    train_dataset = StockDataset(train_data)
    val_dataset = StockDataset(val_data)
    test_dataset = StockDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, val_loader, test_loader



def create_training_data(df_stock, df_news):
    """
    Create training data for stock prediction.
    
    Parameters:
    -----------
    df_stock : pandas.DataFrame
        Stock data with numeric columns scaled
    df_news : pandas.DataFrame
        News data with dates and news token IDs
    
    Returns:
    --------
    list of tuples: Each tuple contains (X, y)
        X: 30 days of numeric stock data
        y: Target value for day 31
    """
    # Ensure dataframes are sorted by date
    df_stock = df_stock.sort_values('date', ascending=True)
    df_news = df_news.sort_values('date', ascending=True)
    
    training_data = []
    
    # Iterate through the dataframe to create training samples
    for i in range(len(df_stock)-2, 29, -1):  # Ensure we have 30 days + 1 target day
        
        # Get the date of the 31st day (target day)
        target_date = df_stock.iloc[i]['date']
        
        # Extract 30 days of numeric data
        X_stock = df_stock.iloc[i-30:i][['open', 'high', 'low', 'close', 'volume', 'adjusted']].values.astype(np.float32)
        X_newsdf = df_news[df_news['date'] < target_date].tail(4)
        X_news_nums = X_newsdf[['sentiment',  'impact',  'positive',  'negative',  'sentiment_score']].values.astype(np.float32)
        X_news_tkids = np.stack(X_newsdf['news_tkids'].values)
        
        # Combine numeric data with news tokens
        X = (X_stock, X_news_nums, X_news_tkids)

        
        # Get target value (31st day's target)
        y = df_stock.iloc[i]['target'].astype(np.float32)
        
        
        training_data.append((X, y))
    
    return training_data

class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X_nums = self.data[idx][0][0]
        X_news_nums = self.data[idx][0][1]
        X_news_tkids = self.data[idx][0][2]
        # because currently only for each day close price is used -> (30, 1)
        y = self.data[idx][1]
        return X_nums, X_news_nums, X_news_tkids, y
    
def custom_collate(batch):
    """
    Custom collate function to handle variable-sized inputs.
    """
    # Assuming `batch` is a list of tuples (X, y), where X can have variable sizes.
    X = [item[0] for item in batch]  # Extract the inputs
    y = [item[1] for item in batch]  # Extract the labels
    
    return X, y  # Return as is (no stacking, just a list of inputs and labels)

    
    
if __name__ == '__main__':
    load_data()