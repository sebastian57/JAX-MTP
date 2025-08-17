import os
import optuna
import jax.numpy as jnp 
import numpy as np   
from motep_jax_train_function import train4 

FIXED_TRAINING_CFG = "Sub_Set.cfg" 
FIXED_LEVEL = 4
FIXED_STEPS_LBFGS = 10                    
FIXED_SPECIES = [0,1]                       
FIXED_PKL_FILE = 'jax_images_data_subset'
FIXED_VAL_PKL_FILE='val_jax_images_data_subset'          

def objective_multi(trial: optuna.trial.Trial):
    """Optuna objective function to wrap the train4 JAX training."""
    level = trial.suggest_int("level", 2, 8, step=2)
    batch_size = trial.suggest_int("batch_size", 5, 10)
    threshold_loss = trial.suggest_float("threshold_loss", 1e-7, 1e-3, log=True)
    min_steps = trial.suggest_int("min_steps", 5, 50)
    lr_start = trial.suggest_float("lr_start", 1e-5, 1e1, log=True)
    transition_steps = trial.suggest_int("transition_steps", 1, 100)
    decay_rate = trial.suggest_float("decay_rate", 0.75, 0.99)
    global_norm_clip = trial.suggest_float("global_norm_clip", 0.01, 2.0) 
    min_dist = trial.suggest_float("min_dist", 0.1, 2.0) 
    max_dist = trial.suggest_float("max_dist", 4.0, 7.0) 
    scaling = trial.suggest_float("scaling", 1e-6, 1.0) 

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Parameters: {trial.params}")


    try:
        epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history = train4(
            training_cfg=FIXED_TRAINING_CFG,
            level=FIXED_LEVEL,
            steps_lbfgs=FIXED_STEPS_LBFGS,
            threshold_loss=threshold_loss, 
            min_steps=min_steps,           
            lr_start=lr_start,             
            transition_steps=transition_steps, 
            decay_rate=decay_rate,         
            global_norm_clip=global_norm_clip,
            min_dist=min_dist,
            max_dist=max_dist,
            scaling=scaling,
            species=FIXED_SPECIES,
            pkl_file=FIXED_PKL_FILE,
            pkl_file_val=FIXED_VAL_PKL_FILE
        )
        
    #try:
    #    epoch_carry, epoch_losses, steps_performed, loss_history, val_loss_history = train5(
    #        training_cfg=FIXED_TRAINING_CFG,
    #        level=FIXED_LEVEL,
    #        batch_size=batch_size,
    #        steps_lbfgs=FIXED_STEPS_LBFGS,
    #        threshold_loss=threshold_loss, 
    #        min_steps=min_steps,           
    #        lr_start=lr_start,             
    #        transition_steps=transition_steps, 
    #        decay_rate=decay_rate,         
    #        global_norm_clip=global_norm_clip,
    #        min_dist=min_dist,
    #        max_dist=max_dist,
    #        scaling=scaling,
    #        species=FIXED_SPECIES,
    #        pkl_file=FIXED_PKL_FILE,
    #        pkl_file_val=FIXED_VAL_PKL_FILE
    #    )
        
        final_loss = val_loss_history[-1]

        total_steps = int(jnp.sum(jnp.array(steps_performed)))

        print(f"  Trial {trial.number} - Total Steps: {total_steps}, Final Loss: {final_loss:.6e}")

        return final_loss, total_steps
    
    except optuna.TrialPruned:
         raise
    except Exception as e:
        print(f"  Trial {trial.number} failed with error: {e}")
        return float('inf'), float('inf')
    

if __name__ == '__main__':
    db_dir = "optuna_studies"
    os.makedirs(db_dir, exist_ok=True)
    
    study_name = f"jax_train4_val_var_level"
    db_filename = f"{study_name}.db"
    db_path = os.path.join(db_dir, db_filename)
    storage_name = f"sqlite:///{db_path}"
    
    print(f"Using Optuna storage: {storage_name}")
        
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        directions=["minimize", "minimize"] 
    )
    
    n_trials_to_run = 10
    try:
        study.optimize(objective_multi, n_trials=n_trials_to_run)
    except KeyboardInterrupt:
         print("Optimization stopped manually.")
    
    print("\n--- Optimization Finished ---")
    
    pareto_optimal_trials = study.best_trials 
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of Pareto optimal trials found: {len(pareto_optimal_trials)}")
    
    print("\nPareto Optimal Trials (Best Trade-offs):")
    for i, trial in enumerate(pareto_optimal_trials):
        loss_val, steps_val = trial.values
        print(f"\n  Trial {trial.number} (Pareto Solution {i+1}):")
        print(f"    Objectives: Loss={loss_val:.6e}, Steps={steps_val}")
        print("    Parameters:")
        for key, value in trial.params.items():
             if isinstance(value, float):
                 print(f"      {key}: {value:.4e}")
             else:
                 print(f"      {key}: {value}")
    
    try:
        fig = optuna.visualization.plot_pareto_front(study, target_names=["Final Loss", "Total Steps"])
        fig.write_image(f"db_dir/{study_name}.pdf")
        fig.show()
    except (ImportError): 
        print("\nInstall plotly for Pareto front visualization (`pip install plotly`)")



