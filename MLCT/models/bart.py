from functools import partial
import pandas as pd
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import rpy2.robjects as robjects
import sys
from rpy2.robjects import numpy2ri, pandas2ri
import numpy as np
from rpy2.robjects.vectors import FloatVector


class BART:
    def __init__(self, t_dim):
        super(BART, self).__init__()
        n_jobs = int(np.rint(5))
        bart = self.install_bart()
        bart.set_bart_machine_num_cores(n_jobs)
        self.bart = bart
        self.t_dim = t_dim
        numpy2ri.activate()
        pandas2ri.activate()

    def install_bart(self):
        robjects.r.options(download_file_method='curl')
        rj = importr("rJava", robject_translations={'.env': 'rj_env'})
        # rj = importr("rJava")
        rj._jinit(parameters="-Xmx16g", force_init=True)
        print("rJava heap size is", np.array(rj._jcall(rj._jnew("java/lang/Runtime"), "J", "maxMemory"))[0] / 1e9,
              "GB.", file=sys.stderr)

        return importr("bartMachine")

    def predict_for_model(self, model, x):
        r = robjects.r
        return np.array(r.predict(self.model, to_data_frame(self.preprocess(x))))

    def train_model(self, args, data):

        tx = data[:, :-1]
        y = data[:, -1].reshape(-1, 1)

        model = self.bart.bartMachine(X=to_data_frame(tx),
                                        y=FloatVector([yy for yy in y]),
                                        mem_cache_for_speed=False,
                                        seed=909,
                                        run_in_sample=False,
                                        alpha=args.bart_alpha_beta[0],
                                        beta=args.bart_alpha_beta[1])
        return model
        #     r = robjects.r
        #     result = np.array(r.predict(model, to_data_frame(tx)))
        #     mse_tmp = ((result - y) ** 2).mean()
        #     mses.append(mse_tmp)
        # mse = np.array(mses).mean()
        # adrf_hat = np.zeros((adrf.shape[0]))
        # # 为了和其他方法估计一致，这里用test的数据
        # for test_id in range(adrf.shape[0]):
        #     t_tmp = adrf[test_id, :self.t_dim].repeat(test_data.shape[0]).reshape((-1, self.t_dim)).detach().numpy()
        #     x_tmp = test_data[:, args.t_dim:-1].detach().numpy()
        #     tx_tmp = test_data[:, :-1].detach().numpy()
        #     y_tmp = test_data[:, -1].reshape(-1, 1).detach().numpy()
        #     # model = self.bart.bartMachine(X=to_data_frame(x_tmp),
        #     #                               y=FloatVector([yy for yy in y_tmp]),
        #     #                               mem_cache_for_speed=False,
        #     #                               seed=909,
        #     #                               run_in_sample=False)
        #     # r = robjects.r
        #     y_hat = r.predict(model, to_data_frame(tx_tmp)).mean()
        #     adrf_hat[test_id] = y_hat
        #     # print(adrf_hat[test_id], adrf[test_id, -1])
        # adrf_mse = ((adrf[:, -1] - adrf_hat) ** 2).mean()
        # return mse, adrf_mse

    def preprocess(self, x):
        if self.with_exposure:
            return np.concatenate([x[0], np.reshape(x[1], (-1, 1)), np.reshape(x[2], (-1, 1))], axis=-1)
        else:
            return np.concatenate([x[0], np.reshape(x[1], (-1, 1))], axis=-1)  # 按行拼接

    def postprocess(self, y):
        return y[:, -1]


def to_data_frame(x, column_names=None):
    if column_names is None:
        column_names = np.arange(x.shape[1])
    return pd.DataFrame(data=x, index=np.arange(x.shape[0]), columns=column_names)
