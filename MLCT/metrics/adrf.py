import numpy as np
from pygam import te
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from models.bart import to_data_frame
import torch

def ADRF(args, model, test_data, adrf):
    adrf_hat = np.zeros((adrf.shape[0]))
    for test_id in range(adrf.shape[0]):

        t_tmp = adrf[test_id, :args.t_dim].repeat(test_data.shape[0]).reshape(-1,1)
        t = test_data[:, :args.t_dim]
        x = test_data[:, args.t_dim:-1]
        tx = np.concatenate((t_tmp, x), axis=1)
        y = test_data[:, -1].reshape(-1, 1)
        if args.model == 'cf':
            base, cf =  model
            base_y = np.array(base.predict(tx))
            r = robjects.r
            out = r.predict(cf, x)
            
            with localconverter(robjects.default_converter + pandas2ri.converter):
                result = robjects.conversion.rpy2py(out)
            y_hat = (np.array(result.values.tolist()).reshape(-1, 1) + base_y).mean()
            adrf_hat[test_id] = y_hat
        if args.model == 'bart':
            r = robjects.r
            y_hat = r.predict(model, to_data_frame(tx)).mean()
            adrf_hat[test_id] = y_hat
        if args.model == 'gps':
            r = robjects.r
            distribution, base = model
            # type_t = np.zeros_like(t)  # 设只有一类t的时候全是0
            # gps = distribution.pdf(type_t)
            gps = distribution.pdf(t_tmp)
            data_frame = pandas2ri.py2rpy(
                to_data_frame(np.column_stack([t_tmp, gps]), column_names=["T", "gps"])
            )
            
            # data_frame = to_data_frame(np.column_stack([type_t, gps]), column_names=["T", "gps"])
            # data_frame = robjects.conversion.py2rpy(data_frame)
            y_hat = r.predict(base, data_frame).mean()
            adrf_hat[test_id] = y_hat
        if args.model == 'cbgps':
            r = robjects.r
            data = to_data_frame(tx, column_names=["T"]+['X'+str(x_idx) for x_idx in range(x.shape[1])])

            y_hat = r.predict(model, newdata=data, type='response').mean()
            # print(y_hat)
            adrf_hat[test_id] = y_hat
        # print(adrf_hat[test_id], adrf[test_id, -1])
        if args.model == 'eb':
            r = robjects.r
            data = to_data_frame(tx, column_names=["T"]+['X'+str(x_idx) for x_idx in range(x.shape[1])])
            y_hat = r.predict(model, newdata=data, type='response').mean()
            adrf_hat[test_id] = y_hat

        if args.model == 'dcows':
            r = robjects.r
            data = to_data_frame(t_tmp, column_names=["T"])
            y_hat = r.predict(model, newdata=data, type='fitp').mean()
            adrf_hat[test_id] = y_hat
        
        if args.model == 'drnet':
            t_tmp = torch.tensor(t_tmp, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            _, y_hat = model(t_tmp, x)
            adrf_hat[test_id] = y_hat.mean().detach().numpy()
        
        if args.model == 'vcnet':
            t_tmp = torch.tensor(t_tmp, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            _, y_hat = model(t_tmp.squeeze(), x)
            adrf_hat[test_id] = y_hat.mean().detach().numpy()
 
    # if args.model == 'dcows':

    #     r = robjects.r
    #     model.np2r(t,'t')
    #     model.np2r(adrf[:,:-1], 'adrf.t')
    #     model.np2r(y, 'y')
    #     adrf_hat = r('weighted_kernel_est(t, y, dcows$weights, adrf.t)$est')
    adrf_mse = ((adrf[:, -1].squeeze() - adrf_hat.squeeze()) ** 2).mean()

    return adrf_mse
