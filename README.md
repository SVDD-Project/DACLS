The Optuna optimization process is launched through the optuna_dcgan.py script using the command python optuna_dcgan.py --dataset ~/dataset/Log3_real/BalancedTraining --validation ~/dataset/Log3/BalancedValidation --n_trials 70.

For a single training run, the run.py script is used with the command python run.py --gan dcgan --dataset ~/dataset/Log3_real/BalancedTraining -log.

The networks currently being developed and tested are implemented in TrainV2.py and dcgan_modelV2.py.
