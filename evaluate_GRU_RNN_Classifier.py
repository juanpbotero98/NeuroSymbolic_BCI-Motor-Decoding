import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Models import GRU_RNN
import nengo  
from util import get_data_BIOCAS, evaluate_classification_model, calculate_trayectory
from data_loader_utils import Batch_Dataset_Discrete
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import argparse

# Main evaluation script
def main(gpu=False):
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
    model_name = "gru_classifier-ls128-sql1000-15epochs"
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

    # train-test split 
    train_FR, test_FR, train_labels, test_labels = train_test_split(FR.T, labels.T,test_size=0.5,shuffle=False) # 50% train, 50% test
    train_FR, test_FR,train_pos, test_pos  = train_test_split(FR.T, cur_pos.T,test_size=0.5,shuffle=False) # 50% train, 50% test

    # train-val split
    train_FR_temp, val_FR, train_labels, val_labels = train_test_split(train_FR, train_labels, test_size=0.25, shuffle=False) # 75% train, 25% validation
    train_FR, val_FR, train_pos, val_pos = train_test_split(train_FR, train_pos, test_size=0.25, shuffle=False) # 75% train, 25% validation

    # Prepare the datasets and data loaders
    # train data 
    train_FR_tensor = torch.tensor(train_FR.T, dtype=torch.float32).to(device)
    train_labels_tensor = torch.tensor(train_labels.T, dtype=torch.long).long().to(device)
    train_dataset = Batch_Dataset_Discrete(train_FR_tensor, train_labels_tensor, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # validation data
    val_FR_tensor = torch.tensor(val_FR.T, dtype=torch.float32).to(device)
    val_labels_tensor = torch.tensor(val_labels.T, dtype=torch.long).long().to(device)
    val_dataset = Batch_Dataset_Discrete(val_FR_tensor, val_labels_tensor, seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test data    
    test_FR_tensor = torch.tensor(test_FR.T, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_labels.T, dtype=torch.long).long().to(device)
    test_dataset = Batch_Dataset_Discrete(test_FR_tensor, test_labels_tensor, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('TRAINING: FR shape= {} | Labels shape= {} | Position shape= {}'.format(train_FR_tensor.shape, train_labels_tensor.shape, train_pos.shape))
    print('VALIDATION: FR shape= {} | Labels shape= {} | Position shape= {}'.format(val_FR_tensor.shape, val_labels_tensor.shape, val_pos.shape))
    print('TEST: FR shape= {} | Labels shape= {} | Position shape= {}'.format(test_FR_tensor.shape, test_labels_tensor.shape, test_pos.shape))

    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    position_gt = {"train": train_pos, "val": val_pos, "test": test_pos}
    
    # Data containers
    all_predictions = []
    all_labels = []
    all_trayectory = []
    phases = []
    all_r2 = []

    # Model evaluation
    for phase, data_loader in data_loaders.items():
        # Get predictions and labels per phase
        accuracy, report, phase_pred, phase_labels = evaluate_classification_model(model, data_loader, device, position_gt[phase])
        print('Phase: {} | Accuracy: {:.2f}%'.format(phase, accuracy * 100))
        print(report)

        # calculate trayectory
        rsquared, trayectory_phase = calculate_trayectory(phase_pred, position_gt[phase], discrete_output=True)

        # Store results
        all_predictions.extend(phase_pred)
        all_labels.extend(phase_labels)
        all_trayectory.extend(trayectory_phase)
        phases.extend([phase]*len(phase_labels))
        all_r2.extend([rsquared]*len(phase_labels))


    all_pos = np.vstack((train_pos, val_pos, test_pos))

    # Save results
    results = np.vstack((all_predictions, all_labels, all_trayectory[:,0].T, all_trayectory[:,1].T,all_pos[:,0].T, all_pos[:,1].T, phases, all_r2)).T
    columns = ["Pred Vel Label", "GT Vel Label", "Pred X-Pos", "Pred Y-Pos", "GT X-Pos", "GT Y-Pos", "Phase", "R2"]	
    results_df = pd.DataFrame(results, columns=columns)
    results_df.to_csv(f"{model_name}_Evaluation_Results.csv", index=False)

    # # Calculate trayectory
    # rsquared, trayectory = calculate_trayectory(all_predictions, test_pos, discrete_output=True)
    # print(f"R-squared: {rsquared:.4f}")
    # # Group trayectory and save as csv
    # pos_results = np.hstack((trayectory, test_pos))
    # columns = ["Predicted Trayectory", "GT Pos"]
    # pos_results_df = pd.DataFrame(pos_results, columns=columns)
    # pos_results_df.to_csv(f"{model_name}_Trayectory_and_GT.csv", index=False)


if __name__ == "__main__":
    # arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=False, help="Use GPU if available")
    args = parser.parse_args()
    main(gpu = args.gpu)
