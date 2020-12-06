import torch

class SGD(object):
    def __init__(self, model, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.model = model
        self.accumulated_grads = dict()

    def optimise(self):
        params = self.model.get_params()
        grad_params = self.model.get_grad_params()

        idx = 0
        for current_layer_vars, current_layer_grads in zip(params, grad_params):
            for current_var, current_grad in zip(current_layer_vars, current_layer_grads):

                if idx not in self.accumulated_grads:
                    self.accumulated_grads[idx] = torch.zeros_like(current_grad)
                old_grad = self.accumulated_grads[idx]

                torch.add(self.momentum * old_grad,
                          self.lr * current_grad,
                          out=old_grad)

                torch.sub(current_var, old_grad, out=current_var)

                idx += 1
