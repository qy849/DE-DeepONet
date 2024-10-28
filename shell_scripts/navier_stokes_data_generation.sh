#!/bin/bash

# bash navier_stokes_data_generation.sh | tee ../logs/navier_stokes_data_generation.log

# Determine the repository path (grandparent directory of this script)
repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"

# Define the paths to configuration and script files
mesh_config_path="$repo_path/configs/navier_stokes/config_data/config_mesh.yaml"
function_space_config_path="$repo_path/configs/navier_stokes/config_data/config_function_space.yaml"
gaussian_random_field_config_path="$repo_path/configs/navier_stokes/config_data/config_gaussian_random_field.yaml"
train_dataset_path="$repo_path/results/navier_stokes/train_dataset"
test_dataset_path="$repo_path/results/navier_stokes/test_dataset"
input_reduced_basis_config_path="$repo_path/configs/navier_stokes/config_data/config_input_reduced_basis.yaml"
input_reduced_basis_path="$repo_path/results/navier_stokes/input_reduced_basis"
output_reduced_basis_config_path="$repo_path/configs/navier_stokes/config_data/config_output_reduced_basis.yaml"
output_reduced_basis_path="$repo_path/results/navier_stokes/output_reduced_basis"
test_dm_path="$repo_path/results/navier_stokes/test_dm"


# Define the paths to python files
GRF_bilaplacian_2D_path="$repo_path/data_generation/probability_measure/GRF_bilaplacian_2D.py"
navier_stokes_path="$repo_path/data_generation/differential_equations/navier_stokes.py"
navier_stokes_evaluation_labels_path="$repo_path/data_generation/evaluation_labels/navier_stokes_evaluation_labels.py"
navier_stokes_plot_input_output_path="$repo_path/data_generation/visualization/navier_stokes_plot_input_output.py"

navier_stokes_ASM_input_reduced_basis_path="$repo_path/data_generation/input_reduced_basis/navier_stokes_ASM_input_reduced_basis.py"
KLE_input_reduced_basis_path="$repo_path/data_generation/input_reduced_basis/KLE_input_reduced_basis.py"
Random_input_reduced_basis_path="$repo_path/data_generation/input_reduced_basis/Random_input_reduced_basis.py"
plot_input_reduced_basis_path="$repo_path/data_generation/visualization/plot_input_reduced_basis.py"

navier_stokes_ASM_output_reduced_basis_path="$repo_path/data_generation/output_reduced_basis/navier_stokes_ASM_output_reduced_basis.py"
navier_stokes_POD_output_reduced_basis_path="$repo_path/data_generation/output_reduced_basis/navier_stokes_POD_output_reduced_basis.py"
navier_stokes_plot_output_reduced_basis_path="$repo_path/data_generation/visualization/navier_stokes_plot_output_reduced_basis.py"

ASM_reduced_inputs_path="$repo_path/data_generation/reduced_inputs/ASM_reduced_inputs.py"
KLE_reduced_inputs_path="$repo_path/data_generation/reduced_inputs/KLE_reduced_inputs.py"

navier_stokes_ASM_reduced_outputs_path="$repo_path/data_generation/reduced_outputs/navier_stokes_ASM_reduced_outputs.py"
navier_stokes_POD_reduced_outputs_path="$repo_path/data_generation/reduced_outputs/navier_stokes_POD_reduced_outputs.py"

input_low_rank_approximations_path="$repo_path/data_generation/low_rank_approximations/input_low_rank_approximations.py"
navier_stokes_output_low_rank_approximations_path="$repo_path/data_generation/low_rank_approximations/navier_stokes_output_low_rank_approximations.py"

navier_stokes_output_reconstruction_error_without_decoder_path="$repo_path/data_generation/output_reconstruction_error/navier_stokes_output_reconstruction_error_without_decoder.py"
navier_stokes_output_reconstruction_error_with_decoder_path="$repo_path/data_generation/output_reconstruction_error/navier_stokes_output_recconstruciton_error_with_decoder.py"
plot_output_reconstruction_error_without_decoder_path="$repo_path/data_generation/visualization/plot_output_reconstruction_error_without_decoder.py"
plot_output_reconstruction_error_with_decoder_path="$repo_path/data_generation/visualization/plot_output_reconstruction_error_with_decoder.py"

navier_stokes_derivative_m_labels_path="$repo_path/data_generation/derivative_labels/navier_stokes_derivative_m_labels.py"
navier_stokes_derivative_x_labels_path="$repo_path/data_generation/derivative_labels/navier_stokes_derivative_x_labels.py"
navier_stokes_reduced_jacobian_labels_path="$repo_path/data_generation/derivative_labels/navier_stokes_reduced_jacobian_labels.py"

GRF_bilaplacian_2D_for_testing_dm_path="$repo_path/data_generation/testing_dm/GRF_bilaplacian_2D_for_testing_dm.py"
navier_stokes_derivative_m_labels_for_testing_dm_path="$repo_path/data_generation/testing_dm/navier_stokes_derivative_m_labels_for_testing_dm.py"
transformation_matrices_path="$repo_path/data_generation/testing_dm/transformation_matrices.py"

compute_dm_direction_vertex_values_path="$repo_path/data_generation/others/compute_dm_direction_vertex_values.py"
compute_output_finite_element_basis_vertex_values_path="$repo_path/data_generation/others/compute_output_finite_element_basis_vertex_values.py"

##############################################################################################################

mpirun -n 64 python "$GRF_bilaplacian_2D_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --train_dataset_path "$train_dataset_path" \
    --test_dataset_path "$test_dataset_path" 

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    mpirun -n 64 python "$navier_stokes_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path" 
done

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    python "$navier_stokes_evaluation_labels_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path" 
done

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    python "$navier_stokes_plot_input_output_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path" 
done

################################################################################################################

mpirun -n 16 python "$navier_stokes_ASM_input_reduced_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --train_dataset_path "$train_dataset_path" \
    --input_reduced_basis_config_path "$input_reduced_basis_config_path" \
    --input_reduced_basis_path "$input_reduced_basis_path"

python "$KLE_input_reduced_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --input_reduced_basis_config_path "$input_reduced_basis_config_path" \
    --input_reduced_basis_path "$input_reduced_basis_path"

mpirun -n 16 python "$Random_input_reduced_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --input_reduced_basis_config_path "$input_reduced_basis_config_path" \
    --input_reduced_basis_path "$input_reduced_basis_path"


python "$plot_input_reduced_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --input_reduced_basis_path "$input_reduced_basis_path"

###############################################################################################################

mpirun -n 16 python "$navier_stokes_ASM_output_reduced_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --train_dataset_path "$train_dataset_path" \
    --output_reduced_basis_config_path "$output_reduced_basis_config_path" \
    --output_reduced_basis_path "$output_reduced_basis_path"

python "$navier_stokes_POD_output_reduced_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --train_dataset_path "$train_dataset_path" \
    --output_reduced_basis_config_path "$output_reduced_basis_config_path" \
    --output_reduced_basis_path "$output_reduced_basis_path"

python "$navier_stokes_plot_output_reduced_basis_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --output_reduced_basis_path "$output_reduced_basis_path"

###############################################################################################################

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    python "$ASM_reduced_inputs_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
        --dataset_path "$dataset_path" \
        --input_reduced_basis_path "$input_reduced_basis_path"
done

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    python "$KLE_reduced_inputs_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
        --dataset_path "$dataset_path" \
        --input_reduced_basis_path "$input_reduced_basis_path"
done

##############################################################################################################

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    python "$navier_stokes_ASM_reduced_outputs_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path" \
        --output_reduced_basis_path "$output_reduced_basis_path"
done

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    python "$navier_stokes_POD_reduced_outputs_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path" \
        --output_reduced_basis_path "$output_reduced_basis_path"
done

#############################################################################################################

input_reduced_basis_names=("ASM" "KLE")
dataset_paths=("$train_dataset_path" "$test_dataset_path")
for input_reduced_basis_name in "${input_reduced_basis_names[@]}"; do
    for num_input_reduced_basis in 1 2 4 8 16; do
        for dataset_path in "${dataset_paths[@]}"; do
            echo "store the data in $dataset_path"
            python "$input_low_rank_approximations_path" \
                --mesh_config_path "$mesh_config_path" \
                --function_space_config_path "$function_space_config_path" \
                --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
                --dataset_path "$dataset_path" \
                --input_reduced_basis_path "$input_reduced_basis_path" \
                --input_reduced_basis_name "$input_reduced_basis_name" \
                --num_input_reduced_basis "$num_input_reduced_basis" 
        done
    done
done


output_reduced_basis_names=("POD" "ASM")
dataset_paths=("$train_dataset_path" "$test_dataset_path")
for output_reduced_basis_name in "${output_reduced_basis_names[@]}"; do
    for num_output_reduced_basis in 64; do
        for dataset_path in "${dataset_paths[@]}"; do
            echo "store the data in $dataset_path"
            python "$navier_stokes_output_low_rank_approximations_path" \
                --mesh_config_path "$mesh_config_path" \
                --function_space_config_path "$function_space_config_path" \
                --dataset_path "$dataset_path" \
                --output_reduced_basis_path "$output_reduced_basis_path" \
                --output_reduced_basis_name "$output_reduced_basis_name" \
                --num_output_reduced_basis "$num_output_reduced_basis"
        done
    done
done

#############################################################################################################

### Optional -- only for analysis of reconstruction error due to input reduction, still using numerical solver ###

# mpirun -n 2 python "$navier_stokes_output_reconstruction_error_without_decoder_path" \
#     --mesh_config_path "$mesh_config_path" \
#     --function_space_config_path "$function_space_config_path" \
#     --train_dataset_path "$train_dataset_path" \
#     --input_reduced_basis_path "$input_reduced_basis_path"

# python "$plot_output_reconstruction_error_without_decoder_path" --input_reduced_basis_path "$input_reduced_basis_path"

# mpirun -n 2 python "$navier_stokes_output_reconstruction_error_with_decoder_path" \
#     --mesh_config_path "$mesh_config_path" \
#     --function_space_config_path "$function_space_config_path" \
#     --train_dataset_path "$train_dataset_path" \
#     --input_reduced_basis_path "$input_reduced_basis_path" \
#     --output_reduced_basis_path "$output_reduced_basis_path" \
#     --output_reduced_basis_name "POD"

# python "$plot_output_reconstruction_error_with_decoder_path" --input_reduced_basis_path "$input_reduced_basis_path" --output_reduced_basis_name "POD"


############################################################################################################

basis_names=("ASM" "KLE" "Random")
dataset_paths=("$train_dataset_path" "$test_dataset_path")
for basis_name in "${basis_names[@]}"; do
    for dataset_path in "${dataset_paths[@]}"; do
        echo "run with $basis_name basis, store the data in $dataset_path"
        mpirun -n 64 python "$navier_stokes_derivative_m_labels_path"\
            --mesh_config_path "$mesh_config_path" \
            --function_space_config_path "$function_space_config_path" \
            --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
            --basis_name "$basis_name" \
            --dataset_path "$dataset_path" \
            --input_reduced_basis_path "$input_reduced_basis_path" \
            --samples_start_index 0 
    done
done

dataset_paths=("$train_dataset_path" "$test_dataset_path")
for dataset_path in "${dataset_paths[@]}"; do
    echo "store the data in $dataset_path"
    mpirun -n 64 python "$navier_stokes_derivative_x_labels_path"\
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dataset_path "$dataset_path" \
        --samples_start_index 0
done


mpirun -n 64 python "$GRF_bilaplacian_2D_for_testing_dm_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --test_dm_path "$test_dm_path" \
    --num_directions 128 \
    --seed 64

mpirun -n 64 python "$navier_stokes_derivative_m_labels_for_testing_dm_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --test_dataset_path "$test_dataset_path" \
    --test_dm_path "$test_dm_path" 

python "$transformation_matrices_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
    --input_reduced_basis_path "$input_reduced_basis_path" \
    --test_dm_path "$test_dm_path"

input_reduced_basis_names=("ASM")
output_reduced_basis_names=("POD" "ASM")
dataset_paths=("$train_dataset_path" "$test_dataset_path")
for input_reduced_basis_name in "${input_reduced_basis_names[@]}"; do
    for output_reduced_basis_name in "${output_reduced_basis_names[@]}"; do
        for dataset_path in "${dataset_paths[@]}"; do
            echo "run with $input_reduced_basis_name input basis and $output_reduced_basis_name output basis, store the data in $dataset_path"
            mpirun -n 64 python "$navier_stokes_reduced_jacobian_labels_path" \
                --mesh_config_path "$mesh_config_path" \
                --function_space_config_path "$function_space_config_path" \
                --gaussian_random_field_config_path "$gaussian_random_field_config_path" \
                --dataset_path "$dataset_path" \
                --input_reduced_basis_path "$input_reduced_basis_path" \
                --output_reduced_basis_path "$output_reduced_basis_path" \
                --input_reduced_basis_name "$input_reduced_basis_name" \
                --output_reduced_basis_name "$output_reduced_basis_name" \
                --num_output_reduced_basis 64
        done
    done
done


basis_names=("ASM" "KLE" "Random")
for basis_name in "${basis_names[@]}"; do
    dm_direction_nodal_values_path="$repo_path/results/navier_stokes/input_reduced_basis/$basis_name/nodal_values.npy"
    dm_direction_vertex_values_path="$repo_path/results/navier_stokes/input_reduced_basis/$basis_name/vertex_values.npy"
    python "$compute_dm_direction_vertex_values_path" \
        --mesh_config_path "$mesh_config_path" \
        --function_space_config_path "$function_space_config_path" \
        --dm_direction_nodal_values_path "$dm_direction_nodal_values_path" \
        --dm_direction_vertex_values_path "$dm_direction_vertex_values_path"
done

dm_direction_nodal_values_path="$repo_path/results/navier_stokes/test_dm/dm_direction_nodal_values.npy"
dm_direction_vertex_values_path="$repo_path/results/navier_stokes/test_dm/dm_direction_vertex_values.npy"
python "$compute_dm_direction_vertex_values_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --dm_direction_nodal_values_path "$dm_direction_nodal_values_path" \
    --dm_direction_vertex_values_path "$dm_direction_vertex_values_path"
    
## Used for computing dm outputs in DINO (should be viewed as one part of the model inference)
mpirun -n 16 python "$compute_output_finite_element_basis_vertex_values_path" \
    --mesh_config_path "$mesh_config_path" \
    --function_space_config_path "$function_space_config_path" \
    --save_path "$output_reduced_basis_path" \
    --problem "navier_stokes" 