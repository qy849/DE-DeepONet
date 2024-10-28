import os

from utils import load_pkl, get_optimizer, get_lr_scheduler

def generate_training_components(args, saved_models_directory):
    model_name = args['model'] 
    model_path = os.path.join(saved_models_directory, f'{model_name}.pkl')
    model = load_pkl(model_path)

    optimizer = get_optimizer(args['optimizer'])
    optimizer_params = {key.split('.')[-1]:value for key, value in args.items() if key.startswith('optimizer_params.')}
    optimizer = optimizer(model.parameters(), **optimizer_params)

    lr_scheduler = get_lr_scheduler(args['lr_scheduler'])
    lr_scheduler_params = {key.split('.')[-1]:value for key, value in args.items() if key.startswith('lr_scheduler_params.')}
    lr_scheduler = lr_scheduler(optimizer, **lr_scheduler_params)

    return model, optimizer, lr_scheduler