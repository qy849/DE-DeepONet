#!/bin/bash

mkdir -p ../results/hyperelasticity

mkdir -p ../results/hyperelasticity/input_reduced_basis
mkdir -p ../results/hyperelasticity/loss_and_error
mkdir -p ../results/hyperelasticity/output_reduced_basis
mkdir -p ../results/hyperelasticity/saved_models
mkdir -p ../results/hyperelasticity/test_dataset
mkdir -p ../results/hyperelasticity/test_dm
mkdir -p ../results/hyperelasticity/train_dataset

mkdir -p ../results/hyperelasticity/input_reduced_basis/ASM
mkdir -p ../results/hyperelasticity/input_reduced_basis/KLE
mkdir -p ../results/hyperelasticity/input_reduced_basis/Random
mkdir -p ../results/hyperelasticity/input_reduced_basis/ASM/figures
mkdir -p ../results/hyperelasticity/input_reduced_basis/KLE/figures
mkdir -p ../results/hyperelasticity/input_reduced_basis/Random/figures

mkdir -p ../results/hyperelasticity/loss_and_error/fno
mkdir -p ../results/hyperelasticity/loss_and_error/deeponet
mkdir -p ../results/hyperelasticity/loss_and_error/dino
mkdir -p ../results/hyperelasticity/loss_and_error/de_deeponet
mkdir -p ../results/hyperelasticity/loss_and_error/dino/ASM_ASM
mkdir -p ../results/hyperelasticity/loss_and_error/dino/ASM_POD
mkdir -p ../results/hyperelasticity/loss_and_error/de_deeponet/ASM
mkdir -p ../results/hyperelasticity/loss_and_error/de_deeponet/KLE

mkdir -p ../results/hyperelasticity/output_reduced_basis/ASM
mkdir -p ../results/hyperelasticity/output_reduced_basis/POD
mkdir -p ../results/hyperelasticity/output_reduced_basis/ASM/figures
mkdir -p ../results/hyperelasticity/output_reduced_basis/POD/figures

mkdir -p ../results/hyperelasticity/saved_models/after_training
mkdir -p ../results/hyperelasticity/saved_models/before_training
mkdir -p ../results/hyperelasticity/saved_models/after_training/fno
mkdir -p ../results/hyperelasticity/saved_models/after_training/deeponet
mkdir -p ../results/hyperelasticity/saved_models/after_training/dino
mkdir -p ../results/hyperelasticity/saved_models/after_training/de_deeponet
mkdir -p ../results/hyperelasticity/saved_models/after_training/dino/ASM_ASM
mkdir -p ../results/hyperelasticity/saved_models/after_training/dino/ASM_POD
mkdir -p ../results/hyperelasticity/saved_models/after_training/de_deeponet/ASM
mkdir -p ../results/hyperelasticity/saved_models/after_training/de_deeponet/KLE
mkdir -p ../results/hyperelasticity/saved_models/before_training/fno
mkdir -p ../results/hyperelasticity/saved_models/before_training/deeponet
mkdir -p ../results/hyperelasticity/saved_models/before_training/dino
mkdir -p ../results/hyperelasticity/saved_models/before_training/de_deeponet
mkdir -p ../results/hyperelasticity/saved_models/before_training/dino/ASM_ASM
mkdir -p ../results/hyperelasticity/saved_models/before_training/dino/ASM_POD
mkdir -p ../results/hyperelasticity/saved_models/before_training/de_deeponet/ASM
mkdir -p ../results/hyperelasticity/saved_models/before_training/de_deeponet/KLE

mkdir -p ../results/hyperelasticity/test_dataset/derivative_labels
mkdir -p ../results/hyperelasticity/test_dataset/figures
mkdir -p ../results/hyperelasticity/test_dataset/input_functions
mkdir -p ../results/hyperelasticity/test_dataset/low_rank_input_functions
mkdir -p ../results/hyperelasticity/test_dataset/low_rank_output_functions
mkdir -p ../results/hyperelasticity/test_dataset/output_functions
mkdir -p ../results/hyperelasticity/test_dataset/reduced_inputs
mkdir -p ../results/hyperelasticity/test_dataset/reduced_outputs
mkdir -p ../results/hyperelasticity/test_dataset/low_rank_input_functions/figures
mkdir -p ../results/hyperelasticity/test_dataset/low_rank_output_functions/figures

mkdir -p ../results/hyperelasticity/train_dataset/derivative_labels
mkdir -p ../results/hyperelasticity/train_dataset/figures
mkdir -p ../results/hyperelasticity/train_dataset/input_functions
mkdir -p ../results/hyperelasticity/train_dataset/low_rank_input_functions
mkdir -p ../results/hyperelasticity/train_dataset/low_rank_output_functions
mkdir -p ../results/hyperelasticity/train_dataset/output_functions
mkdir -p ../results/hyperelasticity/train_dataset/reduced_inputs
mkdir -p ../results/hyperelasticity/train_dataset/reduced_outputs
mkdir -p ../results/hyperelasticity/train_dataset/low_rank_input_functions/figures
mkdir -p ../results/hyperelasticity/train_dataset/low_rank_output_functions/figures