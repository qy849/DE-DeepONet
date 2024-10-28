#!/bin/bash

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"

input_reduced_basis_names=("ASM")
output_reduced_basis_names=("POD")
seeds=(0 1 2 3 4)


for input_reduced_basis_name in "${input_reduced_basis_names[@]}"; do
    for output_reduced_basis_name in "${output_reduced_basis_names[@]}"; do
        echo "model training with encoder as ${input_reduced_basis_name} and decoder as ${output_reduced_basis_name}"
        mesh_config_path="$repo_path/configs/navier_stokes/config_data/config_mesh.yaml"
        function_space_config_path="$repo_path/configs/navier_stokes/config_data/config_function_space.yaml"

        config_models_path="$repo_path/configs/navier_stokes/config_models/dino/${input_reduced_basis_name}_${output_reduced_basis_name}"
        before_training_models_path="$repo_path/results/navier_stokes/saved_models/before_training/dino/${input_reduced_basis_name}_${output_reduced_basis_name}"
        after_training_models_path="$repo_path/results/navier_stokes/saved_models/after_training/dino/${input_reduced_basis_name}_${output_reduced_basis_name}"
        
        train_args_path="$repo_path/configs/navier_stokes/config_training/dino/${input_reduced_basis_name}_${output_reduced_basis_name}/train_args.yaml"
        update_train_args_path="$repo_path/configs/navier_stokes/config_training/dino/${input_reduced_basis_name}_${output_reduced_basis_name}/update_train_args.yaml"
        train_dataset_path="$repo_path/results/navier_stokes/train_dataset"
        
        test_args_path="$repo_path/configs/navier_stokes/config_training/dino/${input_reduced_basis_name}_${output_reduced_basis_name}/test_args.yaml"
        test_dataset_path="$repo_path/results/navier_stokes/test_dataset"
        test_dm_path="$repo_path/results/navier_stokes/test_dm"

        loss_and_error_path="$repo_path/results/navier_stokes/loss_and_error/dino/${input_reduced_basis_name}_${output_reduced_basis_name}"
        
        output_reduced_basis_path="$repo_path/results/navier_stokes/output_reduced_basis"

        dino_path="$repo_path/ml_workflow/dino.py"

        for seed in "${seeds[@]}"; do
            python "$dino_path" \
                --mesh_config_path "$mesh_config_path" \
                --function_space_config_path "$function_space_config_path" \
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
                --input_reduced_basis_name "$input_reduced_basis_name" \
                --output_reduced_basis_name "$output_reduced_basis_name" \
                --output_reduced_basis_path "$output_reduced_basis_path" \
                --require_standardizing_inputs \
                --seed "$seed"
        done
    done
done
