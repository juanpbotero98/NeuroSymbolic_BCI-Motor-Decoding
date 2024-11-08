## Authors: Madeleine Bartlett, 
from pynwb import NWBHDF5IO
import numpy as np
import sparse
from scipy import interpolate
import nengo
nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from scipy.special import legendre
import h5py
import matplotlib.patches as mpatches
import torch
from sklearn.metrics import accuracy_score, classification_report

# ------------ Data Handling Functions -------------- #
def get_data(inputfile):
    """
    The function `get_data` reads data from an NWB file, processes spike times and cursor positions, and
    returns split data based on trial start and stop times.

    :param inputfile: The `inputfile` parameter in the `get_data` function is a file path to an NWB file
    that contains neural data recorded during an experiment. The function reads various data from this
    NWB file, such as trial start and stop times, spike times of neural units, maze conditions, cursor
    :return: The function `get_data(inputfile)` returns two lists: `curs_list` and `spike_list`. These
    lists contain data related to cursor position and spike times, respectively.
    """

    print('Reading data from {}'.format(inputfile))

    with NWBHDF5IO(inputfile, "r") as io:  # Read raw data
        read_nwbfile = io.read()
        start_times = read_nwbfile.trials['start_time'][:]  # Read start time of each trial
        stop_time = read_nwbfile.trials['stop_time'][:]  # Read end time of each trial
        # cue_time = read_nwbfile.intervals['trials']['go_cue_time'][:]
        # move_begint = read_nwbfile.intervals['trials']['move_begins_time'][:] #Begin time of the movement
        # move_endt = read_nwbfile.intervals['trials']['move_ends_time'][:]
        datafile = read_nwbfile.units['spike_times'][:]  # Read spike time of each channel
        targetind = read_nwbfile.intervals['trials']['maze_condition'][:]
        # targetsuc = read_nwbfile.intervals['trials']['correct_reach'][:]
        # cursdir = read_nwbfile.processing['behavior']['Position']['Cursor'].data[:].T
        curstime = read_nwbfile.processing['behavior']['Position']['Cursor'].timestamps[:]
        cursdir = read_nwbfile.processing['behavior']['Position']['Hand'].data[:].T
        # Curser Direction is actually hand position

    datafile[1] = np.hstack((datafile[0], datafile[1]))

    dt = 0.001
    n_ele = 191
    stop_t = stop_time[-1]
    start_t = start_times[0]
    cur_stop_ind = int((curstime[-1] - start_t) // dt)

    res_ = np.zeros((n_ele, (int((stop_t - start_t) // dt)) + 1), dtype=float)
    for i in range(1, len(datafile)):
        spike_time = np.floor((datafile[i] - start_t) // dt).astype(int)
        ele = np.zeros(len(spike_time)).astype(int)

        # spikes = np.ones(len(spike_time)).astype(int)
        coord_array = np.vstack((ele, spike_time))

        s = sparse.COO(coord_array, 1, shape=(1, np.shape(res_)[1]))
        res_[i - 1, :] = res_[i - 1, :] + s.todense()

    res_ = res_[:, :cur_stop_ind]

    # Interpolate the position
    curs_inter = np.array([interpolate_position(cursdir[i], (curstime - start_t) / dt,
                                                np.shape(res_)[1]) for i in range(len(cursdir))])

    # Calculate the velocity
    # curs_vel = calculate_derivative(curs_inter.T, 1).transpose()
    # curs_vel = curs_vel - np.mean(curs_vel, axis=0)

    curs_list = split_data(curs_inter, start_times, stop_time, dt)
    spike_list = split_data(res_, start_times, stop_time, dt)
    print('')

    return curs_list, spike_list, targetind

def get_vel_data(inputfile):
    """
    The function `get_data` reads data from an NWB file, processes spike times and cursor positions, and
    returns split data based on trial start and stop times.

    :param inputfile: The `inputfile` parameter in the `get_data` function is a file path to an NWB file
    that contains neural data recorded during an experiment. The function reads various data from this
    NWB file, such as trial start and stop times, spike times of neural units, maze conditions, cursor
    :return: The function `get_data(inputfile)` returns two lists: `curs_list` and `spike_list`. These
    lists contain data related to cursor position and spike times, respectively.
    """

    with NWBHDF5IO(inputfile,"r") as io: #Read raw data
        read_nwbfile = io.read()
        start_times = read_nwbfile.trials['start_time'][:] #Read start time of each trial
        stop_time = read_nwbfile.trials['stop_time'][:] #Read end time of each trial
        # cue_time = read_nwbfile.intervals['trials']['go_cue_time'][:]
        # move_begint = read_nwbfile.intervals['trials']['move_begins_time'][:] #Begin time of the movement
        # move_endt = read_nwbfile.intervals['trials']['move_ends_time'][:]
        datafile = read_nwbfile.units['spike_times'][:] #Read spike time of each channel
        targetind = read_nwbfile.intervals['trials']['maze_condition'][:]
        # targetsuc = read_nwbfile.intervals['trials']['correct_reach'][:]
        # cursdir = read_nwbfile.processing['behavior']['Position']['Cursor'].data[:].T
        curstime = read_nwbfile.processing['behavior']['Position']['Cursor'].timestamps[:]
        cursdir = read_nwbfile.processing['behavior']['Position']['Hand'].data[:].T
        # Curser Direction is actually hand position

    datafile[1] = np.hstack((datafile[0],datafile[1]))

    dt = 0.001
    n_ele = 191
    stop_t = stop_time[-1]
    start_t = start_times[0]
    cur_stop_ind = int((curstime[-1]-start_t)//dt)

    res_ = np.zeros((n_ele,(int((stop_t-start_t)//dt))+1),dtype=float)
    for i in range(1, len(datafile)):
        spike_time = np.floor((datafile[i] - start_t)//dt).astype(int)
        ele = np.zeros(len(spike_time)).astype(int)

        # spikes = np.ones(len(spike_time)).astype(int)
        coord_array = np.vstack((ele, spike_time))

        s = sparse.COO(coord_array, 1, shape=(1, np.shape(res_)[1]))
        res_[i-1,:] = res_[i-1,:] + s.todense()

    res_ = res_[:, :cur_stop_ind]

    # Interpolate the position
    curs_inter = np.array([interpolate_position(cursdir[i],(curstime-start_t)/dt,
                                                np.shape(res_)[1]) for i in range(len(cursdir))])

    # Calculate the velocity
    curs_vel = calculate_derivative(curs_inter.T, 1).transpose()
    #curs_vel = curs_vel - np.mean(curs_vel, axis=0)

    curs_vel_list = split_data(curs_vel, start_times, stop_time, dt)
    spike_list = split_data(res_, start_times, stop_time, dt)
    print('')

    return curs_vel_list, spike_list

def get_data_BIOCAS(inputfile, discretize_output=False):
    # Load and unpack mat file
    data = h5py.File(inputfile,'r')
    cursor_pos = data['cursor_pos'][:]
    finger_pos = data['finger_pos'][:]
    time = data['t'][:][0]
    target_pos = data['target_pos'][:]
    # Interpolate the position data, from 250 Hz to 1000 Hz to match spike data
    n_ele = 192
    dt = 0.001 # 1 ms --> 1000 Hz resampling
    start_time = time[0]
    stop_time = time[-1]
    exp_length = int((stop_time - start_time) // dt) + 1 
    time_ms = (time-time[0])/dt
    curs_inter = np.array([interpolate_position(cursor_pos[i],time_ms,exp_length) for i in range(len(cursor_pos))])
    
    # Normalize the cursor position
    curs_norm = normalize_to_range(curs_inter,new_min=-1,new_max=1,axis=1)
    
    # Load spike data
    spike_times = data['spikes'][:]
    # for one cell of the spike_times 
    # ref = spike_times[0,0] --> first index == columns, second index == rows
    # spike_times_0_0 = data[ref][:]
    spike_t = []
    for i in range(0,spike_times.shape[1]):
        electrode_data = []
        for j in range(0,spike_times.shape[0]):
            ref = spike_times[j,i]
            spikes = data[ref][:][0]
            if type(spikes) == np.ndarray:	
                electrode_data.append(spikes)

        if electrode_data:
            electrode_data = np.sort(np.concatenate(electrode_data))
            spike_t.append(electrode_data)
        else:
            spike_t.append(np.array([0]))
    # Generate sparse matrix of spike data
    spike_data = np.zeros((n_ele, exp_length))
    for i in range(len(spike_t)):
        if spike_t[i].size > 0:
            spike_time = np.floor((spike_t[i] - start_time) // dt).astype(int) 
            # remove negative spike times. Spike recorded before the start of the cursor data
            spike_time = spike_time[spike_time >= 0]

            # remove spikes recorded after the end of the cursor data
            spike_time = spike_time[spike_time < exp_length]

            # Create sparse matrix
            ele = np.zeros(len(spike_time)).astype(int)
            coord_array = np.vstack((ele, spike_time))
            s = sparse.COO(coord_array, 1, shape=(1, exp_length))
            spike_data[i, :] = s.todense()

    # Calculate the velocity
    #Get the raw speed
    pos_diff = np.diff(curs_norm,axis=1)
    vel_data = pos_diff/dt # pixels per second
    # Prepend zero velocity to match the length of the spike data
    vel_data = np.hstack((np.zeros((2,1)),vel_data))

    if discretize_output:
        vel_data = discretize_velocity(vel_data)

    return curs_norm, spike_data, vel_data

def discretize_velocity(vel_data, vel_tresh=0.81, moving_thresh=0.03): # Thresholds set by 25th and 75th percentile
    # Discretize the velocity data
    # calculate the magnitude and direction of the velocity
    vel_mag = np.linalg.norm(vel_data,axis=0)
    vel_dir = np.arctan2(vel_data[1],vel_data[0])

    # # Create empty array to store discretized labels
    # # 8 directions, 2 speeds + 1 stationary   
    # vel_data = np.zeros((17,np.shape(vel_data)[1]))
    vel_discrete = []

    # Discretize the velocity data
    bins2vect_mapping = {'0-0': 1, '0-1': 2, '45-0':3 , '45-1':4, 
                        '90-0':5, '90-1':6, '135-0':7, '135-1':8, 
                        '180-0':9, '180-1':10, '225-0':11, '225-1':12,
                        '270-0':13, '270-1':14, '315-0':15, '315-1':16}
    
    for mag,dir in zip(vel_mag,vel_dir):
        if mag < moving_thresh:
            # If the magnitude of the velocity is less than the moving threshold, the cursor is considered stationary
            # and the direction is not relevant
            label = 0 # Stationary
            vel_discrete.append(label)
        
        else:
            # Dicretize the direction (8 bins, each 45 degrees)
            dir_deg = np.rad2deg(dir)
            if dir_deg < 0:
                dir_deg += 360
            direction_bin = int(np.floor(dir_deg/45))*45

            # Discretize the speed (2 bins, 0 or 1)
            veolicty_bin = 1 if mag>= vel_tresh else 0

            # Create label vector
            label_idx = bins2vect_mapping[f'{direction_bin}-{veolicty_bin}']
            vel_discrete.append(label_idx)
    return np.array(vel_discrete)
        

def split_data(input, startTime, endTime, dt):
    """
    This Python function takes input data, start and end times, and a time interval, and splits the data
    into segments based on the specified time intervals.

    :param input: input data (np.array) to be split into trials.
    :param startTime: array of trial starting times
    :param endTime: The `endTime` parameter in the `split_data` function represents the end time for
    each data segment. It is used to determine the end index for slicing the input data array
    :param dt: The `dt` parameter in the `split_data` function represents the time interval in seconds
    between each data point. It is used to determine how the data should be split based on the specified
    start and end times
    :return: The function `split_data` returns a list of numpy arrays, where each array contains a
    subset of the input data based on the specified start and end times and time interval.
    """
    res = []
    endTime = (endTime - startTime[0]) // dt
    startTime = (startTime - startTime[0]) // dt
    for i in range(len(startTime)):
        startind = int(startTime[i])
        endind = int(endTime[i])
        res.append(input[:, startind:endind])
    return res


def interpolate_position(position, time, shape):
    """
    The function `interpolate_position` uses linear interpolation to calculate hand positions at
    specified time points based on given position data.

    :param position: Position is a list of coordinates representing the position of a hand or object in
    space. Each coordinate could be a tuple (x, y, z) or a list [x, y, z]
    :param time: Time is a 1-dimensional array representing the time points at which the position values
    are recorded
    :param shape: The `shape` parameter in the `interpolate_position` function represents the number of
    points you want to interpolate between the given `position` values over the specified `time`
    intervals. It determines how finely you want to interpolate the position values
    :return: The function `interpolate_position` returns the interpolated hand position at each time
    step within the specified shape.
    """
    f = interpolate.interp1d(time, position)
    hand_position = f(np.arange(0, shape))
    return hand_position


def calculate_derivative(data, dx):
    """
    Calculate the numerical derivative of an array without changing its shape.

    Parameters:
    data (numpy.ndarray): Input data array.
    dx (float): The spacing between data points (assumed to be uniform).

    Returns:
    numpy.ndarray: The derivative of the input data array with the same shape.
    """
    # Ensure the input is a numpy array
    data = np.asarray(data)

    # Initialize an array to hold the derivative with the same shape as the input
    derivative = np.zeros_like(data)

    # Use central differences for the interior points
    derivative[1:-1] = (data[2:] - data[:-2]) / (2 * dx)

    # Use forward difference for the first point
    derivative[0] = (data[1] - data[0]) / dx

    # Use backward difference for the last point
    derivative[-1] = (data[-1] - data[-2]) / dx

    return derivative


# def evaluate_features(train_X, train_Y, test_X, test_Y, **kwargs):
#     if "models" in kwargs:
#         models = kwargs["models"]
#     else:
#         models = [Naive_model]
#     if "regs" in kwargs:
#         regs = kwargs["regs"]
#     else:
#         regs = [0.1]  # Best reg of default model
#     if "taus" in kwargs:
#         taus = kwargs["taus"]
#     else:
#         taus = [0.7]  # Best tau in Naive model


def normalize_to_range(matrix, new_min, new_max, axis=0):
    def normalize_col(col):
        min_val = np.min(col)
        max_val = np.max(col)
        normalized_col = (col - min_val) / (max_val - min_val)  # Normalize to 0-1 range
        scaled_col = normalized_col * (new_max - new_min) + new_min  # Scale to new range
        return scaled_col

    return np.apply_along_axis(normalize_col, axis, matrix)

def compute_correlation_matrix(projected_data, cursor_data):
    n_A = projected_data.shape[1]
    n_B = cursor_data.shape[1]
    correlation_matrix = np.zeros((n_A, n_B))
    
    for i in range(n_A):
        for j in range(n_B):
            correlation_matrix[i, j] = np.corrcoef(projected_data[:, i], cursor_data[:, j])[0, 1]
    
    plt.figure()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation between Principal Components and Dynamical Components')
    plt.xlabel('Dynamical Components')
    plt.ylabel('Principal Components')
    plt.xticks(range(cursor_data.shape[1]), [f'Dynamic{i+1}' for i in range(cursor_data.shape[1])], rotation=90)
    plt.yticks(range(projected_data.shape[1]), [f'PC{i+1}' for i in range(projected_data.shape[1])])
    plt.grid(False)
    plt.show()
    return correlation_matrix

def evaluate_classification_model(model, data_loader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            batch_size, seq_len, _ = inputs.size()
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size).to(device)
            
            # Process each time step
            for t in range(seq_len):
                input_t = inputs[:, t, :]
                output_t, hidden = model(input_t, hidden)
            
            # Get the predictions for the final time step in the sequence
            _, predicted = torch.max(output_t, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Compute accuracy and detailed classification report
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, zero_division=0)

    # TODO: Calculate the trayectory given the discretized velocity and direction joint labels
    # Class 0: stationary, Class 1: 45 deg 1 pixel, Class 2: 45 deg 2 pixels, ..., Class 16: 315 deg 2 pixels
    # test_init_idx = np.shape(cur_pos)[1]//2
    # innitial_pos = cur_pos[:,test_init_idx]
    
    return accuracy, report

# ------------ Plotting Functions -------------- #
def pred_vs_gt_plot(predictions, ground_truth, r_squared, title=''): #predictions [N,2], ground_truth [N,2]
    ## Plot predictions against actual
    x = np.linspace(-200,200)
    y = np.linspace(-200,200)
    plt.figure()
    plt.suptitle(title)
    plt.subplot(2,1,1)
    plt.scatter(ground_truth[:,0], predictions[:,0], alpha=0.01)
    plt.plot(x,y, ls='--', color='black')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('X values')
    frame1 = plt.gca()
    frame1.axes.set_xticklabels([])
    plt.annotate(f"Test R2: {'{:.2f}'.format(r_squared)}", (150,150))

    plt.subplot(2,1,2)
    plt.scatter(ground_truth[:,1], predictions[:,1], alpha=0.01)
    plt.plot(x,y, ls='--', color='black')
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('Y values')
    plt.annotate(f"Test R2: {'{:.2f}'.format(r_squared)}", (150,150))
    plt.tight_layout()

def determine_velocity_thresholds_and_plot(velocity_magnitudes, stationary_percentile=25, slow_fast_percentile=75):
    # Calculate the thresholds based on specified percentiles
    stationary_threshold = np.percentile(velocity_magnitudes, stationary_percentile)
    slow_fast_threshold = np.percentile(velocity_magnitudes, slow_fast_percentile)
    
    # Create histogram data
    counts, bins = np.histogram(velocity_magnitudes, bins=50)
    
    # Plot histogram with colored bars
    plt.figure(figsize=(10, 6))
    for i in range(len(bins) - 1):
        if bins[i] <= stationary_threshold:
            color = 'red'      # Stationary
        elif bins[i] <= slow_fast_threshold:
            color = 'blue'     # Slow
        else:
            color = 'orange'   # Fast
        plt.bar(bins[i], counts[i], width=bins[i+1] - bins[i], color=color, align='edge')

    # Labels and title
    plt.title("Velocity Magnitude Histogram with Thresholds")
    plt.xlabel("Velocity Magnitude")
    plt.ylabel("Frequency")
    
    # Add legend for color codes
    red_patch = mpatches.Patch(color='red', label='Stationary')
    blue_patch = mpatches.Patch(color='blue', label='Slow')
    orange_patch = mpatches.Patch(color='orange', label='Fast')
    plt.legend(handles=[red_patch, blue_patch, orange_patch])

    plt.show()
    
    return stationary_threshold, slow_fast_threshold


# ------------ Helper Classes -------------- #

# Classes to load and save the weights of a Nengo connection (i.e., save the model)
# taken from: https://github.com/nengo/nengo-extras/issues/35 (i.e., Terry)
# loads a decoder from a file, defaulting to zero if it doesn't exist
class LoadFrom(nengo.solvers.Solver):
    def __init__(self, filename, weights=False):
        super(LoadFrom, self).__init__(weights=weights)
        self.filename = filename

    def __call__(self, A, Y, rng=None, E=None):
        if self.weights:
            shape = (A.shape[1], E.shape[1])
        else:
            shape = (A.shape[1], Y.shape[1])

        try:
            value = np.load(self.filename)
            assert value.shape == shape
        except IOError:
            value = np.zeros(shape)
        return value, {}


# helper to create the LoadFrom solver and the needed probe and do the saving
class WeightSaver(object):
    def __init__(self, connection, filename, sample_every=1.0, weights=False):
        assert isinstance(connection.pre, nengo.Ensemble)
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        self.filename = filename
        connection.solver = LoadFrom(self.filename, weights=weights)
        self.probe = nengo.Probe(connection, 'weights', sample_every=sample_every)
        self.connection = connection

    def save(self, sim):
        np.save(self.filename, sim.data[self.probe][-1].T)


if __name__ == "__main__":
    # Test Churchland data loading function
    # inputfile = "Code V2\Dataset\Jekins_1.nwb"
    # curs_list, spike_list, targetind = get_data(inputfile)

    # Test BIOCAS data loading function
    inputfile = "Dataset\\NHP Reaching Sensorimotor Ephys\\indy_20160407_02.mat"
    cursor_pos, spike_data, vel = get_data_BIOCAS(inputfile, discretize_output=True)

    print("Data loaded successfully!")
    