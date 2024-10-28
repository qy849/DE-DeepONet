## DE-DeepONet

This repo is for the implementation of the derivative-enhanced deep operator network.

### Environment setup

1. Install [Anaconda3](https://docs.continuum.io/anaconda/install)

2. Install FEniCS (version 2019.1.0) and other packages (matplotlib, scipy, jupyer, pytorch) in a newly created conda environment:

```
conda create -n DE-DeepONet -c conda-forge fenics==2019.1.0 matplotlib scipy jupyter pytorch
```

ps: FEniCS and PyTorch should be installed simultaneously to avoid package conflicts. And the installation may take some time (mainly due to the process of solving environments). 

3. Activate the environment 'DE-DeepONet':

```
conda activate DE-DeepONet
```

4. Install `hippylib`: 

```
pip install hippylib  --user
```

5. Set the number of threads as 1 (permanently) so that any program that uses OpenMP (Open Multi-Processing) use only one thread for parallel execution. 

```
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
source ~/.bashrc
```

### How to run the experiments?

Step 1. Create directories for storing the results of the experiments.

```
cd shell_scripts
bash hyperelasticity_results_directories.sh
bash navier_stokes_results_directories.sh
```

Step 2. Generate data and intermediate results

```
bash hyperelasticity_data_generation.sh
bash navier_stokes_data_generation.sh
```

Step 3. Train and test each model. The configurations for model architectures and training hyperparameters are set in the `config_models` and `config_training` directories for each PDE test case. 

```
bash hyperelasticity_model_training_deeponet.sh
bash hyperelasticity_model_training_de_deeponet.sh
bash hyperelasticity_model_training_dino.sh
bash hyperelasticity_model_training_fno.sh

bash navier_stokes_model_training_deeponet.sh
bash navier_stokes_model_training_de_deeponet.sh
bash navier_stokes_model_training_dino.sh
bash navier_stokes_model_training_fno.sh
```

Explanations of some configuration variables: 
 - alpha: float between 0 and 1; moving average parameter in the self-adaptive learning rate annealing algorithm
 - batch_size.branch: int; the batch size for the branch net
 - batch_size.trunk: int or 'all'; the batch size for the trunk net; 'all' means we use all grid points
 - disable_lr_scheduler: true or false; when true, the learing rate scheduler (lr_scheduler_params) are ignored
 - iterations: int; the number of times the model's weights are updated
 - loss_weights.dm: float or null; initial loss weight for the derivative loss (solution-to-parameter); if null, the loss term will be disabled (the same applies to the following two losses) 
 - loss_weights.dx: float or null; initial loss weight for the derivative loss (solution-to-spatial coordinate) [The code still supports training with dx loss which is disabled in the paper revision period.]
 - loss_weights.eval: float or null; initial loss weight for the evaluation loss
 - sampling_fraction: float between 0 and 1; the fraction of grid points evaluated when the model process a batch of functions
 - update_frequency: int; the number of steps we rebalance the loss weights each time
 - dm_directions_name: one option in ['ASM', 'KLE', 'Random']; the name of the test direction when computing the directional derivative in DE-FNO
 - num_test_dm_directions: int; the number of directions used to test the outputs of solution-to-paramter derivative of each model  
 - test_milestones: a list of int; the iteration milestones at which the model is tested on the test dataset

### Citation
If you found this repository useful, you can cite our paper in the following bibtex format:

TBD