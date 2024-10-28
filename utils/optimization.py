import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# see https://pytorch.org/docs/stable/optim.html
def get_optimizer(name):

    if name in ['adadelta', 'Adadelta']:
        return optim.Adadelta
    elif name in ['adagrad', 'Adagrad']:
        return optim.Adagrad
    elif name in ['adam', 'Adam']:
        return optim.Adam
    elif name in ['adamw', 'AdamW']:
        return optim.AdamW
    elif name in ['sparse_adam', 'SparseAdam']:
        return optim.SparseAdam
    elif name in ['adamax', 'Adamax']:
        return optim.Adamax
    elif name in ['asgd', 'ASGD']:
        return optim.ASGD
    elif name in ['lbfgs', 'LBFGS']:
        return optim.LBFGS
    elif name in ['nadam', 'NAdam']:
        return optim.NAdam
    elif name in ['radam', 'RAdam']:
        return optim.RAdam
    elif name in ['rmsprop', 'RMSprop']:
        return optim.RMSprop
    elif name in ['rprop', 'Rprop']:
        return optim.Rprop
    elif name in ['sgd', 'SGD']:
        return optim.SGD
    else:
        raise NotImplementedError


# see https://pytorch.org/docs/stable/optim.html
def get_lr_scheduler(name):

    if name in ['lambda_lr', 'LambdaLR']:
        return lr_scheduler.LambdaLR
    elif name in ['multiplicative_lr', 'MultiplicativeLR']:
        return lr_scheduler.MultiplicativeLR
    elif name in ['step_lr', 'StepLR']:
        return lr_scheduler.StepLR
    elif name in ['multi_step_lr', 'MultiStepLR']:
        return lr_scheduler.MultiStepLR
    elif name in ['constant_lr', 'ConstantLR']:
        return lr_scheduler.ConstantLR
    elif name in ['linear_lr', 'LinearLR']: 
        return lr_scheduler.LinearLR
    elif name in ['exponential_lr', 'ExponentialLR']:
        return lr_scheduler.ExponentialLR
    elif name in ['polynomial_lr', 'PolynomialLR']:
        return lr_scheduler.PolynomialLR
    elif name in ['cosine_annealing_lr', 'CosineAnnealingLR']:
        return lr_scheduler.CosineAnnealingLR
    elif name in ['sequential_lr', 'SequentialLR']:
        return lr_scheduler.SequentialLR
    elif name in ['reduce_lr_on_plateau', 'ReduceLROnPlateau']:
        return lr_scheduler.ReduceLROnPlateau
    elif name in ['cyclic_lr', 'CyclicLR']:
        return lr_scheduler.CyclicLR
    elif name in ['one_cycle_lr', 'OneCycleLR']:
        return lr_scheduler.OneCycleLR
    elif name in ['cosine_annealing_warm_restarts', 'CosineAnnealingWarmRestarts']:
        return lr_scheduler.CosineAnnealingWarmRestarts
    elif name in ['chained_scheduler', 'ChainedScheduler']:
        return lr_scheduler.ChainedScheduler
    else:
        raise NotImplementedError