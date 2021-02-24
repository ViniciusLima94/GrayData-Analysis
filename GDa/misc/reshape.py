import numpy as np 

def reshape_trials(tensor, nT, nt):
    #  Reshape the tensor to have trials and time as two separeted dimension
    #  print(len(tensor.shape))
    if len(tensor.shape) == 1:
        aux = tensor.reshape([nT, nt])
    if len(tensor.shape) == 2:
        aux = tensor.reshape([tensor.shape[0], nT, nt])
    if len(tensor.shape) == 3:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], nT, nt])
    if len(tensor.shape) == 4:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], nT, nt])
    return aux

def reshape_observations(tensor, nT, nt):
    #  Reshape the tensor to have all the trials concatenated in the same dimension
    if len(tensor.shape) == 2:
        aux = tensor.reshape([nT * nt])
    if len(tensor.shape) == 3:
        aux = tensor.reshape([tensor.shape[0], nT * nt])
    if len(tensor.shape) == 4:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], nT * nt])
    if len(tensor.shape) == 5:
        aux = tensor.reshape([tensor.shape[0], tensor.shape[1], tensor.shape[2], nT * nt])
    return aux
