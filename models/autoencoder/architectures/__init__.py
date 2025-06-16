# imports of each arch class and its MODEL_NAME
from .conv_autoencoder16 import ConvAutoencoder16, MODEL_NAME as NAME16
#from .conv_autoencoder32 import ConvAutoencoder32, MODEL_NAME as NAME32
#from .conv_autoencoder32_dropout import ConvAutoencoder32Dropout, MODEL_NAME as NAME32D
#from .conv_autoencoder32_dropout_norm import ConvAutoencoder32DropoutNorm, MODEL_NAME as NAME32DN

# populate central registry
from ..utils import register_model
register_model(NAME16, ConvAutoencoder16)
# register_model(NAME32, ConvAutoencoder32)
# register_model(NAME32D, ConvAutoencoder32Dropout)
# register_model(NAME32DN, ConvAutoencoder32DropoutNorm)
