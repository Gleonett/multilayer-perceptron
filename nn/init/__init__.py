
from nn.init.he import he
from nn.init.xavier import xavier
from nn.init.standard import standard

initializers = {
    "he":       he,
    "xavier":   xavier,
    "standard": standard
}
