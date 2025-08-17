import argparse
import time
import numpy as np
from motep_jax_train_function import train, train_minibatch, mtp, write_mtp_file
from motep_jax_train_import import plot_timing, plot_loss, plot_test
import matplotlib.pyplot as plt
import pickle
import tracemalloc
from jax.experimental.compilation_cache import compilation_cache as cc
import os

import jax
jax.config.update('jax_enable_x64', False)

cc.reset_cache()

#os.environ['JAX_PLATFORM_NAME'] = 'cpu'

def main():
    parser = argparse.ArgumentParser(description="Run MTP Jax training timing test.")
    parser.add_argument("--scaling", type=float, default=1.0,
                        help="rescaling factor for the mtp (should do nothing here tbh)")
    parser.add_argument("--max_dist", type=float, default=5.0,
                        help="Maximum interaction distance for the mtp")
    parser.add_argument("--min_dist", type=float, default=0.5,
                        help="Minimum interaction distance for the mtp")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Size of the individual batches used for mini batching")
    parser.add_argument("--threshold_loss", type=float, default=1e-5,
                        help="Convergence criterium: loss - prev_loss <= threshold_loss")
    parser.add_argument("--min_steps", type=int, default=10,
                        help="Minimum steps performed before convergence accepted")
    parser.add_argument("--lr_start", type=float, default=1e-2,
                        help="Initial learning rate value")
    parser.add_argument("--transition_steps", type=int, default=1,
                        help="After how many steps does the learning rate decay")
    parser.add_argument("--decay_rate", type=float, default=0.99,
                        help="Decay rate per 1 times transition steps")
    parser.add_argument("--global_norm_clip", type=float, default=1e-1,
                        help="Value for clipping gradients")
    parser.add_argument("--training_cfg", type=str, default=None,
                        help="String of the name of the .cfg file for training")
    parser.add_argument("--min_lev", type=int, default=2,
                        help="Minimum level for training (default: 2)")
    parser.add_argument("--max_lev", type=int, default=18,
                        help="Maximum level for training (non-inclusive, default: 18)")
    parser.add_argument("--steps1", type=int, default=1000,
                        help="Number of training steps (default: 1000)")
    parser.add_argument("--steps2", type=int, default=100,
                        help="Number of training steps (default: 100)")
    parser.add_argument("--steps3", type=int, default=100,
                        help="Number of training steps (default: 100)")
    parser.add_argument("--name", type=str, default="timing",
                        help="Output file name without extension (default: timing)")
    parser.add_argument("--folder_name", type=str, default="timing_folder",
                        help="Output folder name without extension (default: timing_folder)")
    parser.add_argument("--train", type=str, default='true',
                        help="str (true or false) to decide if train1 is to be used.")
    parser.add_argument("--train_minibatch", type=str, default='false',
                        help="str (true or false) to decide if train_minibatch is to be used.")
    parser.add_argument("--memory", type=str, default='false',
                        help="str (true or false) to decide if memory is measured.")
    parser.add_argument("--save", type=str, default='true',
                        help="str (true or false) to decide if data is saved.")
    parser.add_argument("--plot", type=str, default='true',
                        help="str (true or false) to decide if data is plotted.")
    parser.add_argument("--pkl_file", type=str, default='jax_images_data',
                        help="Name of the pkl file containing training data.")
    parser.add_argument("--pkl_file_val", type=str, default='val_jax_images_data',
                        help="Name of the pkl file containing non training data.")
    args = parser.parse_args()
    
    # need to move this species definition someday
    levels = np.arange(args.min_lev, args.max_lev, 2)
    species = [0,1] #None #
    elapsed_times = []
    counter = 0
    
    for level in levels:
        file = f'trained_mtps/{level}_trained.mtp'
        start_time = time.time()
        if args.train4 == 'true':
            epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history = train(args.training_cfg, level, args.steps1, args.threshold_loss, args.min_steps, 
                                                                                                args.lr_start, args.transition_steps, args.decay_rate, args.global_norm_clip, 
                                                                                                min_dist = args.min_dist, max_dist = args.max_dist, scaling = args.scaling,
                                                                                                species=species, pkl_file = args.pkl_file, pkl_file_val = args.pkl_file_val) 
            
            params = epoch_carry['params']
        elif args.train5 == 'true':
            epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history = train_minibatch(args.training_cfg, level, args.batch_size, args.steps1, args.threshold_loss, 
                                                                                                args.min_steps, args.lr_start, args.transition_steps, args.decay_rate, 
                                                                                                args.global_norm_clip, min_dist=args.min_dist, max_dist=args.max_dist, scaling=args.scaling, 
                                                                                                species=species, pkl_file = args.pkl_file, pkl_file_val = args.pkl_file_val) 
            params = epoch_carry['params']
            with open('mtp_params_subset', 'wb') as f:
                pickle.dump(params, f)
        else:
            print('No training function specified')
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_times.append(elapsed_time)
        print(epoch_losses)
        print(steps_performed)
        print(elapsed_time)
        counter += 1
        E, F, sigma, real_values = mtp(args.pkl_file_val,level,params)
        
        if args.plot == 'true':
            plot_loss(loss_history, val_loss_history, steps_performed, args.name, args.folder_name, level, min_ind=0, max_ind=-1, save=args.save)
            plot_test(E,F,sigma,real_values,args.name,args.folder_name,level)
            plt.close()
        else:
            print('Data not plotted')
        
        if elapsed_time >= 1200:
            break
    
    if args.plot == 'true':
        plot_timing(levels, elapsed_times, counter, args.name, args.folder_name)
    else:
        print('Data not plotted')
    
    
if __name__ == '__main__':
    import os
    os.makedirs('training_results', exist_ok=True) 
    main()
