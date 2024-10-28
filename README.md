## DE-DeepONet

This repo is for the implementation of the derivative-enhanced deep operator network.

### Environment setup

1. Install [Anaconda3](https://docs.continuum.io/anaconda/install)

2. Install FEniCS (version 2019.1.0) and other packages (matplotlib, scipy, jupyer, pytorch) in a newly created conda environment (here I name it as 'DE-DeepONet') by running the command (Note that these packages should be installed at the same time, otherwise you will probably encounter package conflicts) (Be patient!):

```
conda create -n DE-DeepONet -c conda-forge fenics==2019.1.0 matplotlib scipy jupyter pytorch
```
3. Activate the environment 'DE-DeepONet' by running

```
conda activate DE-DeepONet
```

4. Install `hippylib` using the command: 

```
pip install hippylib  --user
```

5. Set the number of threads as 1 (permanently) so that any program that uses OpenMP (Open Multi-Processing) use only one thread for parallel execution. 

```
echo 'export OMP_NUM_THREADS=1' >> ~/.bashrc
source ~/.bashrc
```

### How to run the experiments?

1. Create directories for storing the results of the experiments.

```
cd shell_scripts
bash hyperelasticity_results_directories.sh
bash navier_stokes_results_directories.sh
```

2. Generate data and intermediate results

```
bash hyperelasticity_data_generation.sh
bash navier_stokes_data_generation.sh
```

3. Train and test each model. The configurations for model architectures and training hyperparameters can be adjusted in the `config_models` and `config_training` directories for each PDE test case. [The code still supports training with dx loss which is disabled in the paper revision period.]

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