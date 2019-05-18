import tensorflow as tf

# # IDE mode
# from tensorflow.python import keras
# from tensorflow.python.keras.layers import Dense, GRU, Dropout, Embedding

# running mode
from tensorflow import keras
from tensorflow.keras.layers import Dense, GRU, Dropout, Embedding

import numpy as np
import time
import pickle

res = pickle.load(open( "data/training_result", "rb" ) )
