import numpy as np
from scipy.stats import norm
from functools import partial
# from sklearn.decomposition import PCA
# from pygam import LinearGAM, s
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, FactorVector, FloatVector, IntVector
import rpy2.robjects as robjects
from rpy2.robjects import Formula, pandas2ri, numpy2ri


# from simulation_low_d import *

class CBGPS:
    def __init__(self, t_dim):
        super(CBGPS, self).__init__()
        self.cbgps = self.install()
        self.pca = None
        self.t_dim = t_dim

        numpy2ri.activate()
        pandas2ri.activate()

    def install(self):
        return importr("CBPS")
    def train_model(self, data):
        t = data[:, :self.t_dim]
        if t.ndim == 1:
            t.reshape(-1, 1)
        # tx = data[:, :-1]
        x = data[:, self.t_dim:-1]
        y = data[:, -1].reshape(-1, 1)
        model = self.build_drf_model(t, x, y)
        return model

    def build_drf_model(self, t, x, y):
        tmp = np.concatenate([x, np.reshape(t, (-1, 1)), np.reshape(y, (-1, 1))], axis=-1)

        data = to_data_frame(tmp, column_names=['X'+str(x) for x in range(tmp.shape[-1] - 2)] + ["T", "Y"])
        # print(data)
        data_frame = pandas2ri.py2rpy(data)
        fit1 = self.cbgps.CBPS(Formula('T ~  ' + '+'.join(data_frame.names[:-2])), 
                data=data_frame, method='exact')
        r = robjects.r
        model = r.glm(Formula('Y ~  T +' + '+'.join(data_frame.names[:-2])),
                     weights = fit1.rx2('fit1$weights'), family = "gaussian", data=data_frame)
        # print(model)
        return model


def to_data_frame(x, column_names=None):
    if column_names is None:
        column_names = np.arange(x.shape[1])
    return pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=column_names)
