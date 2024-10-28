#!/bin/bash

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"

basis_names=("ASM" "KLE")
seeds=(0 1 2 3 4)

for basis_name in "${basis_names[@]}"; do
    mesh_config_path="$repo_path/configs/navier_stokes/config_data/config_mesh.yaml"
    config_models_path="$repo_path/configs/navier_stokes/config_models/de_deeponet/$basis_name"
    before_training_models_path="$repo_path/results/navier_stokes/saved_models/before_training/de_deeponet/$basis_name"
    after_training_models_path="$repo_path/results/navier_stokes/saved_models/after_training/de_deeponet/$basis_name"

    train_args_path="$repo_path/configs/navier_stokes/config_training/de_deeponet/$basis_name/train_args.yaml"
    update_train_args_path="$repo_path/configs/navier_stokes/config_training/de_deeponet/$basis_name/update_train_args.yaml"
    train_dataset_path="$repo_path/results/navier_stokes/train_dataset"

    test_args_path="$repo_path/configs/navier_stokes/config_training/de_deeponet/$basis_name/test_args.yaml"
    test_dataset_path="$repo_path/results/navier_stokes/test_dataset"
    test_dm_path="$repo_path/results/navier_stokes/test_dm"

    loss_and_error_path="$repo_path/results/navier_stokes/loss_and_error/de_deeponet/$basis_name"

    de_deeponet_path="$repo_path/ml_workflow/de_deeponet.py"

    for seed in "${seeds[@]}"; do
        python "$de_deeponet_path" \
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
            --basis_name "$basis_name" \
            --require_standardizing_inputs \
            --require_positional_encoding \
            --seed "$seed"
    done
done
