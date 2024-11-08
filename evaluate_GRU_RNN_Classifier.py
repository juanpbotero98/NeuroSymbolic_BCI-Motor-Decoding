import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Models import GRU_RNN
import nengo  
from util import get_data_BIOCAS, evaluate_classification_model
from data_loader_utils import Batch_Dataset_Discrete
from sklearn.model_selection import train_test_split
import os

# Main evaluation script
def main():
    # Configurations
    input_size = 192
    latent_size = 128
    num_classes = 16
    seq_len = 1000
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data (assuming numpy arrays)
    inputfile = "Dataset\\NHP Reaching Sensorimotor Ephys\\indy_20160407_02.mat"
    cur_pos, spike_data, labels = get_data_BIOCAS(inputfile, discretize_output=True)
    # Calculate instantaneous firing rate from spike data applying synaptic filter
    synapse = nengo.synapses.Lowpass(tau=0.7) # tau is taken from Naive model (L2 reg linear regression) optimal value 
    FR = synapse.filt(spike_data)

    # Split dataset 
    train_FR, test_FR, train_labels, test_labels = train_test_split(FR.T, labels.T, test_size=0.5,shuffle=False) # 50% train, 50% test
    val_FR, train_FR, val_labels, train_labels = train_test_split(train_FR, train_labels, test_size=0.75,shuffle=False) # 25% val, 75% train

    # Prepare test dataset and data loader
    torch.tensor(test_FR, dtype=torch.float32)
    torch.tensor(test_labels, dtype=torch.long)

    test_dataset = Batch_Dataset_Discrete(test_FR, test_labels, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and load trained weights
    model = GRU_RNN(latent_size=latent_size)
    model.init_model(input_size=input_size, output_size=num_classes)
    model.to(device)
    
    # Load pre-trained model weights if available
    traine_model_path = os.path.join(os.getvwd(),'logs','gru_rnn_model.pth')
    model.load_state_dict(torch.load(traine_model_path, map_location=device))

    # Evaluate the model
    accuracy, report = evaluate_classification_model(model, test_loader, device)

    # Print results
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
