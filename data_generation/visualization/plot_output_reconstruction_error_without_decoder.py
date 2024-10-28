import os
import sys
import argparse

import matplotlib.pyplot as plt

repo_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_path)
# print(f'repo path: {repo_path}')

from utils import load_csv, timing # noqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the output reconstruction error without decoder.')
    parser.add_argument('--input_reduced_basis_path', type=str, help='Path to the input reduced basis')
    args = parser.parse_args()
    input_reduced_basis_path = args.input_reduced_basis_path

    print(f'Running: {sys.argv[0]}')

    ASM_error_dict = load_csv(input_reduced_basis_path+'/ASM/output_reconstruction_error_without_decoder.csv')
    KLE_error_dict = load_csv(input_reduced_basis_path+'/KLE/output_reconstruction_error_without_decoder.csv')

    plt.figure(figsize=(8, 6))
    plt.plot(ASM_error_dict['rank'], ASM_error_dict['error'], linestyle='dotted', label='ASM',  marker='s', markersize=9, color='blue', alpha=0.5)
    plt.plot(KLE_error_dict['rank'], KLE_error_dict['error'], linestyle='dotted',label='KLE',  marker='d', markersize=9, color='red', alpha=0.5)
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.xlabel('rank',fontsize=15)
    plt.ylabel('rel-L2-err',fontsize=15, rotation=0, labelpad=30)
    plt.tick_params(axis='both', which='both', labelsize=15)
    plt.title('Ouput reconstruction error', fontsize=18)
    plt.legend(fontsize=15)
    plt.savefig(input_reduced_basis_path+'/output_reconstruction_error.pdf', bbox_inches='tight')
    plt.close()
