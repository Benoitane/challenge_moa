import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

def mean_log_loss(np_list_loss):
    return np.mean(np_list_loss)
