
class BaseLayer(object):

    def __init__(self):
        self.eval_mode = False
        self.saved_for_backward = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def __call__(self, *args):
        if not self.eval_mode:
            self.save_for_backward(*args)
        return self.forward(*args)

    def save_for_backward(self, *args):
        self.saved_for_backward = tuple(arg.clone() for arg in args)
