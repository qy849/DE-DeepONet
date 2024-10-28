from models import *
from utils import save_yaml, extract_pkl_filenames

def generate_nn(args):
    if args['class'] in globals():
        net = globals()[args['class']]
        if not isinstance(net, type):
            raise TypeError(f"'{args['class']}' is not a class.")
        net = net(**args['params'])
    else:
        raise ValueError(f"'{args['class']}' not found in globals().")
    return net


def check_models(yaml_config, yaml_config_path, saved_models_directory):
    available_models = extract_pkl_filenames(saved_models_directory)
    candidate_models = list(set.intersection(set(yaml_config['model']), set(available_models)))
    remaining_models = list(set(yaml_config['model']) - set(available_models))
    if remaining_models:
        print(f'{".".join(remaining_models)} not defined yet.')
        yaml_config['model'] = candidate_models
        save_yaml(yaml_config_path, yaml_config)
        print(f'{yaml_config_path} changed.')

    print(f'candidate models: {" ".join(candidate_models)}')
