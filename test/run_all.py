from test.loss_test import LossTest
from test.layer_test import LayerTest

from nn.layers.linear import __linear_test
from nn.layers.dropout import __dropout_test
from nn.layers.activation.relu import __relu_test
from nn.layers.activation.softmax import __softmax_test

from nn.losses.bce import __bce_test
from nn.losses.mse import __mse_test


if __name__ == '__main__':
    # LAYERS
    __relu_test(LayerTest)
    __linear_test(LayerTest)
    __softmax_test(LayerTest)
    __dropout_test(LayerTest)

    # LOSSES
    __bce_test(LossTest)
    __mse_test(LossTest)
