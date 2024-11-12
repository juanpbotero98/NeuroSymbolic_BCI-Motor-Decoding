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
import argparse
import datetime

def main(epochs= 10, model_path=None, log_dir=None):
    # Configuration
    input_size = 192 # Electrodes
    output_size = 2 # X and Y velocity
    latent_size = 128 # Arbitrary
    batch_size = 32 
    epochs = 15
    learning_rate = 1e-3 # Arbitrary
    seq_len = 1000  # 1s sequence length (arbitrary)

    # Model ID = {date}-GRU-regressor-ls{latent_size}-lr{learning_rate}-bs{batch_size}-sl{seq_len}
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_id = f"{date}-GRU-regressor-ls{latent_size}-lr{learning_rate}-bs{batch_size}-sl{seq_len}-{epochs}epochs"
    log_dir = f"./Training_logs/{model_id}"
    os.makedirs(log_dir, exist_ok=True)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

    # Initialize Model
    model = GRU_RNN(latent_size=latent_size)
    model.init_model(input_size=input_size, output_size=output_size)
    model.to(device)

    # Define Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load model checkpoint if provided
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        print(f"Loaded model checkpoint from epoch {start_epoch} with loss {loss}")
    
    else:
        start_epoch = 0

    # Load data 
    file_name = "indy_20160407_02.mat"
    inputfile = os.path.join("Dataset", "NHP Reaching Sensorimotor Ephys", file_name)
    cur_pos, spike_data, cur_vel = get_data_BIOCAS(inputfile)
    # Calculate instantaneous firing rate from spike data applying synaptic filter
    synapse = nengo.synapses.Lowpass(tau=0.7) # tau is taken from Naive model (L2 reg linear regression) optimal value 
    FR = synapse.filt(spike_data)

    # Split dataset 
    train_FR, test_FR, train_vel, test_vel = train_test_split(FR.T, cur_vel.T, test_size=0.5,shuffle=False) # 50% train, 50% test
    val_FR, train_FR, val_cursor, train_vel = train_test_split(train_FR, train_vel, test_size=0.75,shuffle=False) # 25% val, 75% train
    train_FR_tensor = torch.tensor(train_FR.T, dtype=torch.float32).to(device)
    train_vel_tensor = torch.tensor(train_vel.T, dtype=torch.float32).to(device)

    train_data = Batch_Dataset(train_FR_tensor, train_vel_tensor, seq_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)


    # Training loop
    for epoch in range(epochs-start_epoch):
        epoch += start_epoch
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
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

        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(log_dir, f"gru_rnn_checkpoint.pth"))


    # Close the TensorBoard writer
    writer.close()

    # To monitor training, run this command in the terminal:
    # tensorboard --logdir=./logs

if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    args = parser.parse_args()

    # Run training loop
    main(epochs=args.epochs, model_path=args.model_path)