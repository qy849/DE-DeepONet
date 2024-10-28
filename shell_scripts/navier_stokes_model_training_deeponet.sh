#!/bin/bash

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"

mesh_config_path="$repo_path/configs/navier_stokes/config_data/config_mesh.yaml"

config_models_path="$repo_path/configs/navier_stokes/config_models/deeponet"
before_training_models_path="$repo_path/results/navier_stokes/saved_models/before_training/deeponet"
after_training_models_path="$repo_path/results/navier_stokes/saved_models/after_training/deeponet"

train_args_path="$repo_path/configs/navier_stokes/config_training/deeponet/train_args.yaml"
update_train_args_path="$repo_path/configs/navier_stokes/config_training/deeponet/update_train_args.yaml"
train_dataset_path="$repo_path/results/navier_stokes/train_dataset"

test_args_path="$repo_path/configs/navier_stokes/config_training/deeponet/test_args.yaml"
test_dataset_path="$repo_path/results/navier_stokes/test_dataset"
test_dm_path="$repo_path/results/navier_stokes/test_dm"

loss_and_error_path="$repo_path/results/navier_stokes/loss_and_error/deeponet"

deeponet_path="$repo_path/ml_workflow/deeponet.py"

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
    python "$deeponet_path" \
        --mesh_config_path "$mesh_config_path" \
        --config_models_path "$config_models_path" \
        --before_training_models_path "$before_training_models_path" \
        --after_training_models_path "$after_training_models_path" \
        --train_args_path "$train_args_path" \
        --update_train_args_path "$update_train_args_path" \
        --train_dataset_path "$train_dataset_path" \
        --test_args_path "$test_args_path" \
        --test_dataset_path "$test_dataset_path" \
        --test_dm_path "$test_dm_path" \
        --loss_and_error_path "$loss_and_error_path" \
        --require_positional_encoding \
        --seed "$seed" 
done
