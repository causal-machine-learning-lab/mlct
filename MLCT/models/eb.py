from re import L
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

class EB:
    def __init__(self, t_dim, r_lib_path='/Library/Frameworks/R.framework/Versions/4.1/Resources/library'):
        super(EB, self).__init__()
        self.eb= self.install(r_lib_path)
        self.t_dim = t_dim

        numpy2ri.activate()
        pandas2ri.activate()

    def install(self, r_lib_path):
        importr('survey')
        return importr("entbal", lib_loc=r_lib_path)
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
        print(tmp.shape)
        data = to_data_frame(tmp, column_names=['X'+str(x) for x in range(tmp.shape[-1] - 2)] + ["T", "Y"])
        # print(data)
        data_frame = pandas2ri.py2rpy(data)
        # eb_pars = robjects.ListVector({'exp_type': 'continuous', 'n_moments':3,
        #             'max_iters':1000, 'estimand':'ATE', 'verbose':'FALSE',
        #             'optim_method':'l-bfgs-b', 'bal_tol':1e-8 })
        # fit1 = self.eb.entbal(Formula('T ~  ' + '+'.join(data_frame.names[:-2])), 
        #             data=data_frame,  eb_pars = eb_pars)

        r = robjects.r
        # r.assign('fit1',fit1)
        # wts = r('fit1$wts')
        xt = np.concatenate([x, np.reshape(t, (-1, 1))], axis=-1)
        xt_frame = to_data_frame(xt, column_names=['X'+str(x) for x in range(xt.shape[-1] - 1)] + ["T"])
        r.assign('xt', xt_frame)
        r("eb_pars <- list(exp_type = 'continuous', estimand = 'ATT', n_moments = 3, \
                optim_method = 'L-BFGS-B', verbose = T, opt_constraints = c(-250,250),\
                bal_tol = 1e-8,\
                max_iters = 1000,\
                which_z = 1)")
        r("fit1 = entbal(T~., data=xt, eb_pars=eb_pars)")

        

        # design = r.svydesign(ids=~1, weights=~wts, data = data_frame)
        r.assign('data', data)
        design = r('svydesign(ids=~1, weights=fit1$wts, data=data)')
        model = r.svyglm(Formula('Y ~  T+' + '+'.join(data_frame.names[:-2])), design = design)

        return model


def to_data_frame(x, column_names=None):
    if column_names is None:
        column_names = np.arange(x.shape[1])
    return pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=column_names)
