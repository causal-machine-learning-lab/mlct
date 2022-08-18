import numpy as np
from rpy2.robjects.packages import importr
from sklearn import linear_model
import rpy2.robjects as robjects
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects import numpy2ri


class CausalForest:
    def __init__(self, t_dim, alpha=0.5, r_lib_path='/Library/Frameworks/R.framework/Versions/4.1/Resources/library'):
        super(CausalForest, self).__init__()
        self.bart = None
        grf = self.install_grf(r_lib_path)
        self.grf = grf
        numpy2ri.activate()
        self.t_dim = t_dim
        self.base = linear_model.Ridge(alpha=alpha)


    def install_grf(self, r_lib_path):
        return importr("grf", lib_loc=r_lib_path)

    def fit_grf_model(self, x, t, y):
        return self.grf.causal_forest(x,
                                      FloatVector([float(yy) for yy in y]),
                                      FloatVector([float(tt) for tt in t]), num_trees=3000)

    def train_model(self, data):
        t = data[:, :self.t_dim]
        if t.ndim == 1:
            t.reshape(-1, 1)
        tx = data[:, :-1]
        x = data[:, self.t_dim:-1]
        y = data[:, -1].reshape(-1, 1)


        base = self.base.fit(tx, y)
        xx, yy, tt = np.concatenate([x, x], axis=0), np.concatenate([y, y], axis=0), np.concatenate([t, t], axis=0)
        
        cf = self.fit_grf_model(xx, tt, yy)
        
        return (base, cf)

       