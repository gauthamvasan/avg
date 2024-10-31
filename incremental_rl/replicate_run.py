import time, pickle
import os, copy
import numpy as np

from incremental_rl.utils import set_one_thread
from incremental_rl.hyp_sweep import parse_args, sample_hyper_params, set_algorithm_specific_args, get_training_func


def replicate_run():
    tic = time.time()
    args = parse_args()
    args.seed = np.random.randint(0, 1000000)
    
    args = sample_hyper_params(args)

    if args.env == "mountain_car_continuous":
        args.gamma = 1

    if args.env in ["Walker2d-v4", "Hopper-v4"]:
        args.alpha_lr = 0.6
    
    if args.env == "reacher_hard":
        args.gamma = 0.95

    args.experiment_dir = "./"
    os.makedirs(args.results_dir, exist_ok=True)
    pkl_fpath = os.path.join(args.results_dir, "./{}_{}_{}_seed-{}.pkl".format(args.env, args.algo, args.hyp_seed, args.seed))

    if args.no_entropy:
        args.alpha_lr = 0
        pkl_fpath = os.path.join(args.results_dir, "./{}_{}_ne_{}_seed-{}.pkl".format(args.env, args.algo, args.hyp_seed, args.seed))

    # Set custom algorithm variant specific arguments
    set_algorithm_specific_args(args)

    # Torch shenanigans fix
    set_one_thread() 

    # Save hyper-parameters and config info
    hyperparams_dict = vars(args)
    hyperparams_dict["device"] = str(hyperparams_dict["device"])
    pkl_data = {'args': hyperparams_dict}


    run_args = copy.deepcopy(args)
    run_args.seed = args.seed
    runner = get_training_func(run_args.algo)
    ep_steps, rets = runner(run_args)

    ### Saving data
    data = np.zeros((2, len(rets)))
    data[0] = ep_steps
    data[1] = rets
    pkl_data[args.seed] = {'returns': data, 'N': sum(ep_steps), 'R': np.mean(rets)}
    print(f"Seed {args.seed} ended with mean return {np.mean(rets)} in {sum(ep_steps)} steps.")
    with open(pkl_fpath, "wb") as handle:
        pickle.dump(pkl_data, handle)
    
    print("Total time for {} on {}: {:.2f}".format(args.algo, args.env, time.time()-tic))


if __name__ == '__main__':
    replicate_run()
