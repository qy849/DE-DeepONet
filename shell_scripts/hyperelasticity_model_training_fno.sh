#!/bin/bash

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"

mesh_config_path="$repo_path/configs/hyperelasticity/config_data/config_mesh.yaml"
config_models_path="$repo_path/configs/hyperelasticity/config_models/fno"
before_training_models_path="$repo_path/results/hyperelasticity/saved_models/before_training/fno"
after_training_models_path="$repo_path/results/hyperelasticity/saved_models/after_training/fno"
train_dm_direction_path="$repo_path/results/hyperelasticity/input_reduced_basis"

train_args_path="$repo_path/configs/hyperelasticity/config_training/fno/train_args.yaml"
update_train_args_path="$repo_path/configs/hyperelasticity/config_training/fno/update_train_args.yaml"
train_dataset_path="$repo_path/results/hyperelasticity/train_dataset"

test_args_path="$repo_path/configs/hyperelasticity/config_training/fno/test_args.yaml"
test_dataset_path="$repo_path/results/hyperelasticity/test_dataset"
test_dm_path="$repo_path/results/hyperelasticity/test_dm"

loss_and_error_path="$repo_path/results/hyperelasticity/loss_and_error/fno"

fno_path="$repo_path/ml_workflow/fno.py"

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
    python "$fno_path" \
        --mesh_config_path "$mesh_config_path" \
        --config_models_path "$config_models_path" \
        --before_training_models_path "$before_training_models_path" \
        --after_training_models_path "$after_training_models_path" \
        --train_args_path "$train_args_path" \
        --update_train_args_path "$update_train_args_path" \
        --train_dataset_path "$train_dataset_path" \
        --train_dm_direction_path "$train_dm_direction_path" \
        --test_args_path "$test_args_path" \
        --test_dataset_path "$test_dataset_path" \
        --test_dm_path "$test_dm_path" \
        --loss_and_error_path "$loss_and_error_path" \
        --seed "$seed" 
done