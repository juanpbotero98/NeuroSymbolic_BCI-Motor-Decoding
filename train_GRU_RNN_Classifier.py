import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
from Models import GRU_RNN
import nengo  
from util import get_data_BIOCAS
from data_loader_utils import Batch_Dataset_Discrete
from sklearn.model_selection import train_test_split

# Configuration
input_size = 192
num_classes = 17
latent_size = 128
batch_size = 32
seq_len = 1000  # Sequence length of 1 second at 1 kHz
epochs = 3
learning_rate = 1e-3
log_dir = "./logs-classifier"
os.makedirs(log_dir, exist_ok=True)

# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# Initialize Model
model = GRU_RNN(latent_size=latent_size)
model.init_model(input_size=input_size, output_size=num_classes)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load data 
inputfile = "Dataset\\NHP Reaching Sensorimotor Ephys\\indy_20160407_02.mat"
cur_pos, spike_data, labels = get_data_BIOCAS(inputfile, discretize_output=True)
# Calculate instantaneous firing rate from spike data applying synaptic filter
synapse = nengo.synapses.Lowpass(tau=0.7) # tau is taken from Naive model (L2 reg linear regression) optimal value 
FR = synapse.filt(spike_data)

# Split dataset 
train_FR, test_FR, train_labels, test_labels = train_test_split(FR.T, labels.T, test_size=0.5,shuffle=False) # 50% train, 50% test
val_FR, train_FR, val_labels, train_labels = train_test_split(train_FR, train_labels, test_size=0.75,shuffle=False) # 25% val, 75% train
train_FR_tensor = torch.tensor(train_FR.T, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels.T).long()

# Prepare Dataset and DataLoader
train_data = Batch_Dataset_Discrete(train_FR_tensor, train_labels_tensor, seq_len)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Training loop
global_step = 0  # Initialize a global step counter for batch logging

for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
        inputs, targets = batch  # inputs: (batch_size, seq_len, input_size), targets: (batch_size, seq_len)
        batch_size, seq_len, _ = inputs.size()
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size)
        
        # Process each time step
        optimizer.zero_grad()
        loss = 0.0
        batch_correct = 0  # Track correct predictions per batch
        batch_total = 0    # Track total predictions per batch
        for t in range(seq_len):
            input_t = inputs[:, t, :]
            target_t = targets[:, t]
            
            # Forward pass
            output_t, hidden = model(input_t, hidden)  # output_t: (batch_size, num_classes)
            loss += criterion(output_t, target_t)
            
            # Calculate predictions for accuracy
            _, predicted = torch.max(output_t, 1)
            batch_correct += (predicted == target_t).sum().item()
            batch_total += target_t.size(0)
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss and correct predictions for the epoch
        epoch_loss += loss.item() / seq_len
        correct += batch_correct
        total += batch_total
        
        # Log batch loss and accuracy to TensorBoard
        batch_accuracy = 100 * batch_correct / batch_total
        writer.add_scalar("Loss/train_batch", loss.item() / seq_len, global_step)
        writer.add_scalar("Accuracy/train_batch", batch_accuracy, global_step)
        
        global_step += 1
    
    # Average loss per batch for the epoch
    epoch_loss /= len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    # Log epoch loss and accuracy
    writer.add_scalar("Loss/train_epoch", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train_epoch", epoch_accuracy, epoch)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    # Save epoch-wise training log to JSON if desired
    with open(os.path.join(log_dir, "training_log.json"), "a") as f:
        f.write(json.dumps({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_accuracy}) + "\n")

# Save model
torch.save(model.state_dict(), os.path.join(log_dir, "gru_rnn_model.pth"))

# Close the TensorBoard writer
writer.close()