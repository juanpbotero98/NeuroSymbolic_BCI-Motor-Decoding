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
import argparse
import datetime

def main(epochs= 10, model_path=None, log_dir=None, load_model=False):
    # Configuration
    input_size = 192
    num_classes = 17
    latent_size = 128
    # Training parameters
    batch_size = 32
    seq_len = 100  # Sequence length of 1 second at 1 kHz
    stride = 1
    epochs = 15
    learning_rate = 1e-3

    # Model ID = date-GRU-classifier-ls{latent_size}-lr{learning_rate}-bs{batch_size}-sl{seq_len}
    if log_dir is None:
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_id = f"{date}-GRU-classifier-ls{latent_size}-lr{learning_rate}-bs{batch_size}-sl{seq_len}--{epochs}epochs-0.5dropout"
        log_dir = f"./Training_logs/{model_id}" 
    else: 
        model_id = log_dir.split("/")[-1]
        date = model_id.split("-")[0]
    os.makedirs(log_dir, exist_ok=True)


    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))

    # Initialize Model
    model = GRU_RNN(latent_size=latent_size)
    model.init_model(input_size=input_size, output_size=num_classes)
    model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) # L2 regularization	

    # Load model checkpoint if provided
    if load_model:
        model_path = os.path.join(log_dir, "gru_rnn_checkpoint.pth")
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
    inputfile = os.path.join("Dataset", file_name)
    # Check if file exists
    print(os.path.isfile(inputfile))
    cur_pos, spike_data, labels = get_data_BIOCAS(inputfile, discretize_output=True)
    # Calculate instantaneous firing rate from spike data applying synaptic filter
    synapse = nengo.synapses.Lowpass(tau=0.7) # tau is taken from Naive model (L2 reg linear regression) optimal value 
    FR = synapse.filt(spike_data)

    # Split dataset 
    train_FR, test_FR, train_labels, test_labels = train_test_split(FR.T, labels.T, test_size=0.5,shuffle=False) # 50% train, 50% test
    val_FR, train_FR, val_labels, train_labels = train_test_split(train_FR, train_labels, test_size=0.75,shuffle=False) # 25% val, 75% train
    train_FR_tensor = torch.tensor(train_FR.T, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_labels.T).long().to(device)

    # Prepare Dataset and DataLoader
    train_data = Batch_Dataset_Discrete(train_FR_tensor, train_labels_tensor, seq_len)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Training loop
    global_step = 0  # Initialize a global step counter for batch logging

    for epoch in range(epochs-start_epoch):
        epoch += start_epoch
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Model: {model_id} - Epoch: {epoch + 1}/{epochs}")):
            inputs, targets = batch  # inputs: (batch_size, seq_len, input_size), targets: (batch_size, seq_len)
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
            batch_size, seq_len, _ = inputs.size()
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size).to(device)
            
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

        # Save epoch-wise training log to JSON
        with open(os.path.join(log_dir, "training_log.json"), "a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_accuracy}) + "\n")

        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, os.path.join(log_dir, f"gru_rnn_checkpoint.pth"))

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__": 
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("--load_model", type=bool, default=False, help="Load model checkpoint")
    parser.add_argument("--log_dir", type=str, default=None, help="Path to log directory")
    args = parser.parse_args()

    # Run training loop
    main(epochs=args.epochs, load_model=args.load_model, log_dir=args.log_dir)