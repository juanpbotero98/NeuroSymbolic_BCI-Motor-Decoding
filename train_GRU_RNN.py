import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os
import json
from Models import GRU_RNN
import nengo  
from util import get_data_BIOCAS
from data_loader_utils import Batch_Dataset
from sklearn.model_selection import train_test_split


# Configuration
input_size = 192 # Electrodes
output_size = 2 # X and Y velocity
latent_size = 128 # Arbitrary
batch_size = 32 
epochs = 3
learning_rate = 1e-3 # Arbitrary
seq_len = 1000  # 1s sequence length (arbitrary)
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# Initialize Model
model = GRU_RNN(latent_size=latent_size)
model.init_model(input_size=input_size, output_size=output_size)

# Define Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Load data 
inputfile = "Dataset\\NHP Reaching Sensorimotor Ephys\\indy_20160407_02.mat"
cur_pos, spike_data, cur_vel = get_data_BIOCAS(inputfile)
# Calculate instantaneous firing rate from spike data applying synaptic filter
synapse = nengo.synapses.Lowpass(tau=0.7) # tau is taken from Naive model (L2 reg linear regression) optimal value 
FR = synapse.filt(spike_data)

# Split dataset 
train_FR, test_FR, train_vel, test_vel = train_test_split(FR.T, cur_vel.T, test_size=0.5,shuffle=False) # 50% train, 50% test
val_FR, train_FR, val_cursor, train_vel = train_test_split(train_FR, train_vel, test_size=0.75,shuffle=False) # 25% val, 75% train
train_FR_tensor = torch.tensor(train_FR.T, dtype=torch.float32)
train_vel_tensor = torch.tensor(train_vel.T, dtype=torch.float32)

train_data = Batch_Dataset(train_FR_tensor, train_vel_tensor, seq_len)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


# Training loop
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs, targets = batch
        batch_size, seq_len, _ = inputs.size()
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size)
        
        # Process each time step
        optimizer.zero_grad()
        loss = 0.0
        for t in range(seq_len):
            input_t = inputs[:, t, :]
            target_t = targets[:, t, :]
            
            # Forward pass
            output_t, hidden = model(input_t, hidden)
            loss += criterion(output_t, target_t)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += loss.item() / seq_len
    
    # Average loss per batch
    epoch_loss /= len(train_loader)
    
    # Log the loss to TensorBoard
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

    # Save training loss to JSON file for additional logging if desired
    with open(os.path.join(log_dir, "training_log.json"), "a") as f:
        f.write(json.dumps({"epoch": epoch + 1, "loss": epoch_loss}) + "\n")

# Save model
torch.save(model.state_dict(), os.path.join(log_dir, "gru_rnn_model.pth"))

# Close the TensorBoard writer
writer.close()

# To monitor training, run this command in the terminal:
# tensorboard --logdir=./logs
