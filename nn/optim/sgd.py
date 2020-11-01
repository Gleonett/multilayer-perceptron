import torch

def sgd_momentum(variables, gradients, learning_rate, state):
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            old_grad = state['accumulated_grads'].setdefault(var_index, torch.zeros_like(current_grad))

            torch.add(1 * old_grad, learning_rate * current_grad, out=old_grad)

            current_var -= old_grad
            var_index += 1