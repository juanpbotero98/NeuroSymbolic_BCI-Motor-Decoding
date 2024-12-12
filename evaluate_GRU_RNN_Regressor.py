import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from Models import GRU_RNN
import nengo  
from util import get_data_BIOCAS, evaluate_regression_model, calculate_trayectory
from data_loader_utils import Batch_Dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import argparse
# Main evaluation script
def main(model_name, gpu=True, decoded_var = 'vel'):
    # Configurations
    input_size = 192
    latent_size = 128
    output_size = 2
    seq_len = 1000
    batch_size = 32
    
    # Initialize the model and load trained weights
    model = GRU_RNN(latent_size=latent_size)
    model.init_model(input_size, output_size=output_size)
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
    cur_pos, spike_data, cur_vel = get_data_BIOCAS(inputfile, discretize_output=False)
    # Calculate instantaneous firing rate from spike data applying synaptic filter
    synapse = nengo.synapses.Lowpass(tau=0.7) # tau is taken from Naive model (L2 reg linear regression) optimal value 
    FR = synapse.filt(spike_data)

    # train-test split 
    train_FR, test_FR, train_vel, test_vel = train_test_split(FR.T, cur_vel.T,test_size=0.5,shuffle=False) # 50% train, 50% test
    train_FR, test_FR,train_pos, test_pos  = train_test_split(FR.T, cur_pos.T,test_size=0.5,shuffle=False) # 50% train, 50% test

    # train-val split
    train_FR_temp, val_FR, train_vel, val_vel = train_test_split(train_FR, train_vel, test_size=0.25, shuffle=False) # 75% train, 25% validation
    train_FR, val_FR, train_pos, val_pos = train_test_split(train_FR, train_pos, test_size=0.25, shuffle=False) # 75% train, 25% validation
    # Determine the gt signal
    if decoded_var == 'pos':
        train_gt = train_pos
        val_gt = val_pos
        test_gt = test_pos
    
    elif decoded_var == 'vel':
        train_gt = train_vel
        val_gt = val_vel
        test_gt = test_vel
        
    # Prepare the datasets and data loaders
    # Crop the first 3k samples from all data for short debugging
    # train_FR = train_FR[:3011]
    # train_gt = train_gt[:3011]
    # train_pos = train_pos[:3011]
    # val_FR = val_FR[:3011]
    # val_gt = val_gt[:3011]
    # val_pos = val_pos[:3011]
    # test_FR = test_FR[:3011]
    # test_gt = test_gt[:3011]
    # test_pos = test_pos[:3011]
    # Drop the last samples to make the data divisible by the sequence length
    train_FR = train_FR[:-(train_FR.shape[0] % seq_len)]
    train_gt = train_gt[:-(train_gt.shape[0] % seq_len)]
    val_FR = val_FR[:-(val_FR.shape[0] % seq_len)]
    val_gt = val_gt[:-(val_gt.shape[0] % seq_len)]
    test_FR = test_FR[:-(test_FR.shape[0] % seq_len)]
    test_gt = test_gt[:-(test_gt.shape[0] % seq_len)]
    train_pos = train_pos[:-(train_pos.shape[0] % seq_len)]
    val_pos = val_pos[:-(val_pos.shape[0] % seq_len)]
    test_pos = test_pos[:-(test_pos.shape[0] % seq_len)]

    # train data 
    train_FR_tensor = torch.tensor(train_FR.T, dtype=torch.float32).to(device)
    train_gt_tensor = torch.tensor(train_gt.T, dtype=torch.long).long().to(device)
    train_dataset = Batch_Dataset(train_FR_tensor, train_gt_tensor, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # validation data
    val_FR_tensor = torch.tensor(val_FR.T, dtype=torch.float32).to(device)
    val_gt_tensor = torch.tensor(val_gt.T, dtype=torch.long).long().to(device)
    val_dataset = Batch_Dataset(val_FR_tensor, val_gt_tensor, seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test data    
    test_FR_tensor = torch.tensor(test_FR.T, dtype=torch.float32).to(device)
    test_gt_tensor = torch.tensor(test_gt.T, dtype=torch.long).long().to(device)
    test_dataset = Batch_Dataset(test_FR_tensor, test_gt_tensor, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('TRAINING: FR shape= {} | GT shape= {} | Position shape= {}'.format(train_FR_tensor.shape, train_gt_tensor.shape, train_pos.shape))
    print('VALIDATION: FR shape= {} | GT shape= {} | Position shape= {}'.format(val_FR_tensor.shape, val_gt_tensor.shape, val_pos.shape))
    print('TEST: FR shape= {} | GT shape= {} | Position shape= {}'.format(test_FR_tensor.shape, test_gt_tensor.shape, test_pos.shape))

    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    position_gt = {"train": train_pos, "val": val_pos, "test": test_pos}

    # Data containers
    all_predictions_X = []
    all_predictions_Y = []
    all_gt_X = []
    all_gt_Y = []
    all_trayectory_X = []
    all_trayectory_Y = []
    phases = []
    r2_pred = []
    r2_trayectory = []

    for phase, data_loader in data_loaders.items():

        # Evaluate the model
        r_squared_pred, pred_var, gt_var = evaluate_regression_model(model, data_loader, device)
        print('Phase: {} | R-squared: {}'.format(phase, r_squared_pred))

        # Calculate trayectory
        if decoded_var == 'vel':
            trayectory_phase, r_squared_tray = calculate_trayectory(pred_var, test_pos, discrete_output=False)
        else: 
            trayectory_phase = pred_var
            r_squared_tray = r_squared_pred
        
        # Store results
        all_predictions_X.extend(pred_var[:,0])
        all_predictions_Y.extend(pred_var[:,1])
        all_gt_X.extend(gt_var[:,0])
        all_gt_Y.extend(gt_var[:,1])
        all_trayectory_X.extend(trayectory_phase[:,0])
        all_trayectory_Y.extend(trayectory_phase[:,1])
        r2_pred.extend([r_squared_pred]*len(gt_var))
        r2_trayectory.extend([r_squared_tray]*len(gt_var))
        phases.extend([phase]*len(gt_var))
    
    # Remove the first seq_len-1 samples from the pos data to match the length of the predictions
    train_pos = train_pos[seq_len-1:]
    val_pos = val_pos[seq_len-1:]
    test_pos = test_pos[seq_len-1:]
    all_pos = np.vstack((train_pos, val_pos, test_pos))
    all_pos_X = all_pos[:,0]
    all_pos_Y = all_pos[:,1]
    


    # Group trayectory and save as csv
    pos_results = np.vstack((all_predictions_X, all_predictions_Y, all_gt_X, all_gt_Y, all_trayectory_X, all_trayectory_Y,all_pos_X, all_pos_Y, phases, r2_pred, r2_trayectory)).T
    columns = ["Predicted {} X".format(decoded_var), "Predicted {} Y".format(decoded_var), "GT {} X".format(decoded_var), "GT {} X".format(decoded_var), "Pos X", "Pos Y", "GT Pos X", "GT Pos Y", "Phase", "R2 Pred", "R2 Trayectory"]
    pos_results_df = pd.DataFrame(pos_results, columns=columns)
    pos_results_df.to_csv(f"{model_name}_Trayectory_and_GT.csv", index=False)

    # # Print results
    # print(f"Test Accuracy: {accuracy * 100:.2f}%")
    # print("Classification Report:")
    # print(report)
    # print(f"R-squared: {rsquared:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=False, help="Use GPU if available")
    parser.add_argument("--model_name", type=str, default='gru_regressor_pos-ls128-sql100-15epochs', help="Name of the model to evaluate")
    parser.add_argument("--decoded_var", type=str, default='pos', help="Variable to decode: 'pos' or 'vel'")
    args = parser.parse_args()
    main(args.model_name, args.gpu, args.decoded_var)
