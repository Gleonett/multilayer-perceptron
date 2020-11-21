from nn.preprocessing.standard_scale import StandardScale
from nn.preprocessing.minmax_scale import MinMaxScale

scalers = {
    "standard": StandardScale,
    "minmax": MinMaxScale,
}
