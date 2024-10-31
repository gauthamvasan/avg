import torch
import argparse
import os
import copy
import pickle
import time
import numpy as np
from incremental_rl.utils import set_one_thread


def parse_args():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--env', required=True, type=str, help="e.g., 'ball_in_cup', 'sparse_reacher', 'Hopper-v2' ")
    parser.add_argument('--use_image', action='store_true')
    parser.add_argument('--seed', default=0, type=int, help="Seed for random number generator")
    parser.add_argument('--hyp_seed', required=True, type=int, help="Seed for hyper-parameter generation")       
    parser.add_argument('--start_seed', default=0, type=int, help="Seed for random number generator")
    parser.add_argument('--N', default=200000, type=int, help="# timesteps for the run")    
    # Choice of algorithm
    parser.add_argument('--algo', required=True, type=str, help="Algorithm choices: ['oac', 'avg']")    
    parser.add_argument('--no_entropy', default=False, action='store_true', help="Entropy coefficient is zero")
    # SAC hyper-parameters
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--bootstrap_terminal', default=0, type=int, help="Bootstrap on terminal state")
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--update_every', default=1, type=int)
    parser.add_argument('--update_epochs', default=1, type=int)
    parser.add_argument('--update_critic_target_every', default=1, type=int)
    # MLP params
    parser.add_argument('--nn_activation', default="leaky_relu", type=str, help="Activations for neural network")
    parser.add_argument('--nhid_layers', default=2, type=int)
    parser.add_argument('--nhid_actor', default=256, type=int)
    parser.add_argument('--nhid_critic', default=256, type=int)
    # ReDo
    parser.add_argument('--tau', default=0.025, type=float, help="Threshold")
    parser.add_argument('--redo_period', default=1000, type=int, help="ReDo recycling period")    
    # L2 regularizers
    parser.add_argument('--l2_actor', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--l2_critic', default=0, type=float, help="L2 Regularization")    
    parser.add_argument('--l2_action', default=0, type=float, help="Action penalty squared loss")
    # Miscellaneous
    parser.add_argument('--checkpoint', default=10000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--results_dir', default="./results", type=str, help="Location to store results")
    parser.add_argument('--device', default="cpu", type=str)    
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--do_not_save', action='store_true', default=True)
    parser.add_argument('--wandb_mode', default='disabled', type=str, help="Either online, offline, or disabled")
    parser.add_argument('--description', default='', type=str)
    # Abaltions args
    parser.add_argument('--normalize_obs', action='store_true', default=False)    
    parser.add_argument('--pnorm', action='store_true', default=False)
    parser.add_argument('--scaled_td', action='store_true', default=False)
    # Eval
    parser.add_argument('--n_eval', default=0, type=int, help="Number of eval episodes")
    parser.add_argument('--debug', action='store_true', default=False)
    # Number of seeds for each hyp_seed
    parser.add_argument('--n_seeds', default=10, type=int)
    args = parser.parse_args()
    
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")   

    return args


def get_training_func(algo):
    if 'avg' in algo:
        if algo == "avg_tq":
            from incremental_rl.avg_target_q import main    
        else:
            from incremental_rl.avg_ablation import main
    elif algo == "sac":
        from incremental_rl.SAC.sac import main
    elif 'iac' in algo:
        from incremental_rl.norm_incremental_actor_critic import main
    elif "isac" in algo:
        from incremental_rl.norm_isac import main
    else:
        raise Exception(f"{algo} not found!")

    return main


def sample_hyper_params(args):
    rng = np.random.RandomState(seed=args.hyp_seed)

    # Step size
    exponent = rng.choice([4, 5, 6])
    args.actor_lr = rng.choice(np.arange(1, 100)) * (1/np.power(10, exponent))
    exponent = rng.choice([4, 5, 6])
    args.critic_lr = rng.choice(np.arange(1, 100)) * (1/np.power(10, exponent))
    
    # Beta for Adam
    args.beta1 = rng.choice([0., 0.9])
    args.betas = [args.beta1, 0.999]
    
    # Entropy
    exponent = rng.choice([1, 2, 3, 4, 5])
    args.alpha_lr = rng.choice(np.arange(1, 10)) * (1/np.power(10, exponent))
    
    # Discount factor
    args.gamma = rng.choice([0.95, 0.97, 0.99, 0.995, 1])

    # MLP architecture    
    args.actor_nn_params = {
        'mlp': {
            'hidden_sizes': [args.nhid_actor] * args.nhid_layers,
            'activation': args.nn_activation,
        }
    }
    args.critic_nn_params = {
        'mlp': {
            'hidden_sizes': [args.nhid_critic] * args.nhid_layers,
            'activation': args.nn_activation,
        }
    }

    return args


def set_algorithm_specific_args(args):
    #### For SAC variants
    if args.algo == "sac_a":        
        args.replay_buffer_capacity = 100000; args.batch_size = 1
    elif args.algo == "sac_b":        
        args.replay_buffer_capacity = args.batch_size = 256
    elif args.algo == "sac_c":        
        args.replay_buffer_capacity = 256; args.batch_size = 1  
    elif args.algo == "isac":
        args.pnorm = False; args.normalize_obs = False; args.scaled_td = False
    elif args.algo in ["avg_pnorm", "iac_pnorm"]:
        args.pnorm = True; args.normalize_obs = False; args.scaled_td = False
    elif args.algo in ["avg_norm_obs", "iac_norm_obs"]:
        args.pnorm = False; args.normalize_obs = True; args.scaled_td = False
    elif args.algo in ["avg_norm_obs_pnorm", "iac_norm_obs_pnorm"]:
        args.pnorm = True; args.normalize_obs = True; args.scaled_td = False
    elif args.algo in ["avg_scaled_td", "iac_scaled_td"]:
        args.pnorm = False; args.normalize_obs = False; args.scaled_td = True
    elif args.algo in ["avg", "iac_all", "isac_all"]:
        args.pnorm = True; args.normalize_obs = True; args.scaled_td = True
    elif args.algo in ["avg_norm_obs_scaled_td", "iac_norm_obs_scaled_td"]:
        args.pnorm = False; args.normalize_obs = True; args.scaled_td = True
    elif args.algo in ["avg_pnorm_scaled_td", "iac_pnorm_scaled_td"]:
        args.pnorm = True; args.normalize_obs = False; args.scaled_td = True

def random_search():
    tic = time.time()
    args = parse_args()
    
    args = sample_hyper_params(args)

    if args.env == "mountain_car_continuous":
        args.gamma = 1

    args.experiment_dir = "./"
    os.makedirs(args.results_dir, exist_ok=True)
    pkl_fpath = os.path.join(args.results_dir, "./{}_{}_{}.pkl".format(args.env, args.algo, args.hyp_seed))

    if args.no_entropy:
        args.alpha_lr = 0
        pkl_fpath = os.path.join(args.results_dir, "./{}_{}_ne_{}.pkl".format(args.env, args.algo, args.hyp_seed))

    # Set custom algorithm variant specific arguments
    set_algorithm_specific_args(args)

    # Torch shenanigans fix
    set_one_thread() 

    # Save hyper-parameters and config info
    hyperparams_dict = vars(args)
    hyperparams_dict["device"] = str(hyperparams_dict["device"])
    pkl_data = {'args': hyperparams_dict}

    ret_list = []
    ep_lens = []
    for seed in range(args.start_seed, args.start_seed+args.n_seeds):
        run_args = copy.deepcopy(args)
        run_args.seed = seed
        runner = get_training_func(run_args.algo)
        ep_steps, rets = runner(run_args)

        ### Saving data
        data = np.zeros((2, len(rets)))
        data[0] = ep_steps
        data[1] = rets
        pkl_data[seed] = {'returns': data, 'N': sum(ep_steps), 'R': np.mean(rets)}
        if sum(ep_steps) >= args.N:
            ret_list.append(np.mean(rets))
            ep_lens.append(np.mean(ep_steps))
        print(f"Seed {seed} ended with mean return {np.mean(rets)} in {sum(ep_steps)} steps.")

        # Partial save. This should make it easier to resume failed experiments.
        with open(pkl_fpath, "wb") as handle:
            pickle.dump(pkl_data, handle)
    
    # Compute mean and std _only_ if all seeds are present
    if len(ret_list) < args.n_seeds:
        pkl_data["mean"] = None
        pkl_data["std"] = None
        pkl_data["std_err"] = None
        pkl_data["steps_to_goal"] = None
        pkl_data["stg_std_err"] = None
    else:
        pkl_data["mean"] = np.mean(ret_list)
        pkl_data["std"] = np.std(ret_list)
        pkl_data["std_err"] = np.std(ret_list)/np.sqrt(len(ret_list))
        pkl_data["steps_to_goal"] = np.mean(ep_lens)
        pkl_data["stg_std_err"] = np.std(ep_lens)/np.sqrt(len(ep_lens))
    
    # Final pickle dump
    with open(pkl_fpath, "wb") as handle:
        pickle.dump(pkl_data, handle)
    
    print("Total time for {} seeds of {} on {}: {:.2f}".format(
        args.n_seeds, args.algo, args.env, time.time()-tic))


if __name__ == '__main__':
    random_search()
