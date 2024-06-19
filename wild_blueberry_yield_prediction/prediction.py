import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesRegressor

def get_prediction(data, model):
    """
    Predicting the yield for a given datapoint / instance
    """
    return model.predict(data) #the output of the model, that is ext_reg_model.predict(x_test_scaled)