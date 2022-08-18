import numpy as np
from pygam import te
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from models.bart import to_data_frame
import torch

def IDRF(args, model, idrf):
    t = idrf[:, :args.t_dim]
    x = idrf[:,  args.t_dim:-1]
    tx = idrf[:, :-1]
    y = idrf[:, -1].reshape(-1, 1)
       
    if args.model == 'cf':
        base, cf =  model
        base_y = np.array(base.predict(tx))
        r = robjects.r
        out = r.predict(cf, x)
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            result = robjects.conversion.rpy2py(out)
        y_hat = (np.array(result.values.tolist()).reshape(-1, 1) + base_y)

    if args.model == 'bart':
        r = robjects.r
        y_hat = r.predict(model, to_data_frame(tx))

    if args.model == 'gps':
        r = robjects.r
        distribution, base = model
        # type_t = np.zeros_like(t)  # 设只有一类t的时候全是0
        # gps = distribution.pdf(type_t)
        gps = distribution.pdf(t)
        data_frame = pandas2ri.py2rpy(
            to_data_frame(np.column_stack([t, gps]), column_names=["T", "gps"])
        )
        
        # data_frame = to_data_frame(np.column_stack([type_t, gps]), column_names=["T", "gps"])
        # data_frame = robjects.conversion.py2rpy(data_frame)
        y_hat = r.predict(base, data_frame)

    if args.model == 'cbgps':
        r = robjects.r
        data = to_data_frame(tx, column_names=["T"]+['X'+str(x_idx) for x_idx in range(x.shape[1])])

        y_hat = r.predict(model, newdata=data, type='response')

    if args.model == 'eb':
        r = robjects.r
        data = to_data_frame(tx, column_names=["T"]+['X'+str(x_idx) for x_idx in range(x.shape[1])])
        y_hat = r.predict(model, newdata=data, type='response')

    if args.model == 'dcows':
        r = robjects.r
        data = to_data_frame(t, column_names=["t"])
        y_hat = r.predict(model, newdata=data, type='fitp')
        
    if args.model == 'drnet':
        t = torch.tensor(t, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        _, y_hat = model(t, x)
        y_hat = y_hat.detach().numpy()
        
    if args.model == 'vcnet':
        t = torch.tensor(t, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        _, y_hat = model(t.squeeze(), x)
        y_hat = y_hat.detach().numpy()

 
    idrf_mse = ((y.squeeze() - y_hat.squeeze()) ** 2).mean()

    return idrf_mse
