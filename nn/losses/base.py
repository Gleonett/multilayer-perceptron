

class BaseLoss(object):

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)
