import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Models import GRU_RNN
import nengo  
from util import get_data_BIOCAS, evaluate_regression_model
from data_loader_utils import Batch_Dataset_Discrete
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# Main evaluation script
def main(model_name, gpu=True, decoded_var = 'pos'):
    # Configurations
    input_size = 192
    latent_size = 128
    num_classes = 17
    seq_len = 1000
    batch_size = 32
    
    # Initialize the model and load trained weights
    model = GRU_RNN(latent_size=latent_size)
    model.init_model(input_size=input_size, output_size=num_classes)
    # Check if GPU is available
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: {}".format(torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        device = torch.device("cpu")    
    
    model.to(device)
    
    # Load pre-trained model weights if available
    trained_model_path = os.path.join(os.getcwd(),'Trained_Models',f'{model_name}.pth')
    checkpoint = torch.load(trained_model_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    

    # Load test data
    file_name = "indy_20160407_02.mat"
    inputfile = os.path.join("Dataset", file_name)
    cur_pos, spike_data, labels = get_data_BIOCAS(inputfile, discretize_output=True)
    # Calculate instantaneous firing rate from spike data applying synaptic filter
    synapse = nengo.synapses.Lowpass(tau=0.7) # tau is taken from Naive model (L2 reg linear regression) optimal value 
    FR = synapse.filt(spike_data)

    # Split dataset 
    train_FR, test_FR, train_labels, test_labels, train_pos, test_pos = train_test_split(FR.T, labels.T, cur_pos.T,test_size=0.5,shuffle=False) # 50% train, 50% test
    val_FR, train_FR, val_labels, train_labels, train_pos, test_pos = train_test_split(train_FR, train_labels, train_pos,test_size=0.75,shuffle=False) # 25% val, 75% train

    # Prepare test dataset and data loader
    test_FR_tensor = torch.tensor(test_FR.T, dtype=torch.float32).to(device)
    test_pos_tensor = torch.tensor(test_pos.T, dtype=torch.long).long().to(device)
    print(test_FR_tensor.shape)  
    print(test_pos_tensor.shape)

    test_dataset = Batch_Dataset_Discrete(test_FR_tensor, test_pos_tensor, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(len(test_dataset)) 
    # Evaluate the model
    accuracy, report, trayectory, rsquared, all_predictions, all_labels = evaluate_regression_model(model, test_loader, device, test_pos)
    # Save results
    # Group predictions and labels and save as csv
    results = np.hstack((all_predictions, all_labels))
    columns = ["Predicted", "True"]
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(f"{model_name}_Predicted_and_GT.csv", index=False)
    # Group trayectory and save as csv
    pos_results = np.hstack((trayectory, test_pos))
    columns = ["Predicted Trayectory", "True"]
    pos_results_df = pd.DataFrame(pos_results, columns=columns)
    pos_results_df.to_csv(f"{model_name}_Trayectory_and_GT.csv", index=False)

    # Print results
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(report)
    print(f"R-squared: {rsquared:.4f}")


if __name__ == "__main__":
    main()
