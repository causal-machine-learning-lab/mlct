from re import L
import numpy as np
from scipy.stats import norm
from functools import partial
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, FactorVector, FloatVector, IntVector
import rpy2.robjects as robjects
from rpy2.robjects import Formula, pandas2ri, numpy2ri

class DCOWS:
    def __init__(self, t_dim, r_lib_path='/Library/Frameworks/R.framework/Versions/4.1/Resources/library') -> None:
        super(DCOWS, self).__init__()
        self.dcows = self.install(r_lib_path)
        self.t_dim = t_dim
        self.weights = None

        numpy2ri.activate()
        pandas2ri.activate()

    def install(self, r_lib_path):
        importr("locfit", lib_loc=r_lib_path)

        return importr("independenceWeights", lib_loc=r_lib_path)

    def train_model(self, data):
        t = data[:, :self.t_dim]
        if t.ndim == 1:
            t.reshape(-1, 1)
        # tx = data[:, :-1]
        x = data[:, self.t_dim:-1]
        y = data[:, -1].reshape(-1, 1)
        model = self.build_drf_model(t, x, y)
        return model

    def np2r(self, B, B_str):
        numpy2ri.activate()
        nr,nc = B.shape
        Br = robjects.r.matrix(B, nrow=nr, ncol=nc)
        robjects.r.assign(B_str, Br)
        
    def build_drf_model(self, t, x, y):
        self.np2r(t, 't')
        self.np2r(x, 'x')
        self.np2r(y, 'y')
        ty = np.concatenate([t, y], axis=-1)
        ty_frame = to_data_frame(ty, column_names=["T","Y"])
        r = robjects.r
        r.assign('ty', ty_frame)
        r("dcows <- independence_weights(t, x)")
        model = r("locfit::locfit(y ~ locfit::lp(t), weights = dcows$weights, data = ty)")
        return model

def to_data_frame(x, column_names=None):
    if column_names is None:
        column_names = np.arange(x.shape[1])
    return pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=column_names)