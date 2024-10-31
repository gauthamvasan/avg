import torch
import os, json, subprocess

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def typecast_data(x):
    """ Convert int, bool and numpy array to appropriate tensors """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
    if isinstance(x, int) or isinstance(x, float):
        x = torch.tensor([x], dtype=torch.float32)    
    if isinstance(x, bool):
        x = torch.tensor([x], dtype=torch.int)
    return x


def human_format_numbers(num, use_float=False):
    # Make human readable short-forms for large numbers
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if use_float:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 


def set_one_thread():
    '''
    N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
    Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def save_returns(rets, ep_lens, save_path):
    """ Save learning curve data as a numpy text file 

    Args:
        rets (list/array): A list or array of episodic returns
        ep_lens (list/array):  A list or array of episodic length
        savepath (str): Save path
    """
    data = np.zeros((2, len(rets)))
    data[0] = ep_lens
    data[1] = rets
    np.savetxt(save_path, data)


def orthogonal_weight_init(m):
    """ Orthogonal weight initialization for neural networks

    Args:
        m (_type_): _description_
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class NpEncoder(json.JSONEncoder):
    """ 
    JSON does not like Numpy elements. Convert to native python datatypes for json dump.  
    Ref: https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_args(args, save_path):
    """ Save hyper-parameters as a json file """
    hyperparas_dict = vars(args)
    hyperparas_dict["device"] = str(hyperparas_dict["device"])
    json.dump(hyperparas_dict, open(save_path, 'w'), indent=4, cls=NpEncoder)


def get_git_hash():
    try:
        # Run git command to get the hash of the current commit
        hash_output = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        # Decode the byte output to string
        git_hash = hash_output.decode('utf-8')
        return git_hash
    except Exception as e:
        print("Error:", e)
        return None


def smoothed_curve(returns, ep_lens, x_tick=5000, window_len=5000):
    """
    Args:
        returns: 1-D numpy array with episodic returs
        ep_lens: 1-D numpy array with episodic returs
        x_tick (int): Bin size
        window_len (int): Length of averaging window
    Returns:
        A numpy array
    """
    rets = []
    x = []
    cum_episode_lengths = np.cumsum(ep_lens)

    if cum_episode_lengths[-1] >= x_tick:
        y = cum_episode_lengths[-1] + 1
        steps_show = np.arange(x_tick, y, x_tick)

        for i in range(len(steps_show)):
            rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_len)) *
                                     (cum_episode_lengths < x_tick * (i + 1))]
            if rets_in_window.any():
                rets.append(np.mean(rets_in_window))
                x.append((i+1) * x_tick)

    return np.array(rets), np.array(x)


def learning_curve(rets, ep_lens, save_path, x_tick=10000, window_len=10000):
    if len(rets) > 0:
        plot_rets, plot_x = smoothed_curve(np.array(rets), np.array(ep_lens), x_tick=x_tick, window_len=window_len)
        if len(plot_rets):
            plt.clf()
            plt.plot(plot_x, plot_rets)
            plt.pause(0.001)
            plt.savefig(save_path, dpi=200)
            