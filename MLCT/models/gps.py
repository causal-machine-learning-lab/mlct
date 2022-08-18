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

class GPS:
    def __init__(self, t_dim):
        super(GPS, self).__init__()
        self.gps = self.install_grf()
        self.pca = None
        self.t_dim = t_dim

        numpy2ri.activate()
        pandas2ri.activate()

    def install_grf(self):
        return importr("causaldrf")
    def train_model(self, data):
        t = data[:, :self.t_dim]
        if t.ndim == 1:
            t.reshape(-1, 1)
        # tx = data[:, :-1]
        x = data[:, self.t_dim:-1]
        y = data[:, -1].reshape(-1, 1)
        distribution, model = self.build_drf_model(t, x, y)
        return distribution, model

    def build_drf_model(self, t, x, y):
        tmp = np.concatenate([x, np.reshape(t, (-1, 1)), np.reshape(y, (-1, 1))], axis=-1)
        data = to_data_frame(tmp, column_names=['X'+str(x) for x in range(tmp.shape[-1] - 2)] + ["T", "Y"])
        print(data)
        data_frame = pandas2ri.py2rpy(data)

        result = self.gps.hi_est(Y="Y",
                                 treat="T",
                                 treat_formula=Formula('T ~ ' + '+'.join(data_frame.names[:-2])),
                                #  treat_formula='T ~ ' + '+'.join(data_frame.names[:-2]),
                                 outcome_formula=Formula('Y ~ T + I(T^2) + gps + T * gps'),
                                 data=data_frame,
                                 grid_val=FloatVector([float(tt) for tt in np.linspace(0,1, 256)]),
                                 treat_mod="Normal",
                                 link_function="inverse")  # link_function is not used with treat_mod = "Normal".
        treatment_model, model = result[1], result[2]
        fitted_values = treatment_model.rx2('fitted.values')
        distribution = norm(np.mean(fitted_values), np.std(fitted_values))  # 拟合正态分布
        return distribution, model


def to_data_frame(x, column_names=None):
    if column_names is None:
        column_names = np.arange(x.shape[1])
    return pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=column_names)
