from torch.utils.data import Dataset

class Batch_Dataset(Dataset):
    # convert X_train and y_train into sequences of time steps for batch processing in the model, 
    # using a sliding window approach to generate sequences for each batch.
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.size(1) - self.seq_len + 1  # Number of possible sequences

    def __getitem__(self, idx):
        # Extract a sequence of length `seq_len` for X and y starting at `idx`
        x_seq = self.X[:, idx : idx + self.seq_len].T  # Shape (seq_len, 192)
        y_seq = self.y[:, idx : idx + self.seq_len].T  # Shape (seq_len, 2)
        return x_seq, y_seq
    
class Batch_Dataset_Discrete(Dataset):
    def __init__(self, X, y, seq_len, stride=1):
        """
        Args:
            X: Input data, shape (features, timesteps).
            y: Labels, shape (timesteps,).
            seq_len: Length of each sequence.
            stride: Step size between consecutive sequences.
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.stride = stride

    def __len__(self):
        # Number of possible sequences considering the stride
        return (self.X.size(1) - self.seq_len) // self.stride + 1

    def __getitem__(self, idx):
        # Calculate the actual starting index based on the stride
        start_idx = idx * self.stride
        # Extract a sequence of length `seq_len` for X and y starting at `start_idx`
        x_seq = self.X[:, start_idx : start_idx + self.seq_len].T  # Shape (seq_len, features)
        y_seq = self.y[start_idx : start_idx + self.seq_len]  # Shape (seq_len,)
        return x_seq, y_seq

