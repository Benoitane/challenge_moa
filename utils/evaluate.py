import pandas as pd
import numpy as np
from sklearn.metrics import log_loss

def mean_log_loss(np_list_loss):
    return np.mean(np_list_loss)

def train_over_targets(features,targets,model,naive,list_pos_rate):
    models_loss = []
    model.fit(features, targets[:,1])
    naive.fit(features, targets[:,1])
    for i in range(targets.shape[1]):
        if list_pos_rate[i] >= 0.01:
            Y_pred = model.predict(features)
            models_loss.append(log_loss(targets[:,i], Y_pred))
        if list_pos_rate[i] < 0.01:
            Y_pred = naive.predict(features)
            models_loss.append(log_loss(targets[:,i], Y_pred))
    return models_loss
