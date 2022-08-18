import numpy as np
import matplotlib.pyplot as plt

def ADRF(args, A, nobs = 2000, MX1 =  -0.5, MX2  = 1, MX3 = 0.3, A_effect = True):
    A_mean =  np.mean(A, axis=1).squeeze()
    truth = - 1.5 * A_mean - 15 
    truth /= 50

    for i in range(5,args.x_dim):
        if i % 2 == 0:
            truth += i/100
    truth /= 4
    return truth

# def IDRF(args, A, X, nobs = 2000, MX1 =  -0.5, MX2  = 1, MX3 = 0.3, A_effect = True):
#     Cnum = ((MX1+3)**2+1) + 2*((MX2-25)**2+1)
#     A_mean =  np.mean(A, axis=1).squeeze()

#     Y = - 1.5 * A_mean - 15 + (X[:+3)**2 + 2 * (X2-25)**2 + X3 - Cnum + np.random.normal(scale=1,size=nobs)
#     Y /= 50

#     for i in range(5,args.x_dim):
#         if i % 2 == 0:
#             Y += X[:,i].squeeze()
    
#     Y = Y.reshape(-1,1)
#     Y /= 4
#     return Y
def simulate(args, seed = 1,  nobs = 2000, MX1 =  -0.5, MX2  = 1, MX3 = 0.3, A_effect = True):

    X1 = np.random.normal(MX1, 1, nobs)
    X2 = np.random.normal(MX2, 1, nobs)
    X3 = np.random.normal(0,1, nobs)
    X4 = np.random.normal(MX2, 1, nobs)
    X5 = np.random.binomial(1,MX3,nobs)
    
    Z1 = np.exp(X1/2)
    Z2 = (X2/(1+np.exp(X1))) + 10
    Z3 = (X1 * X3 / 25) + 0.6
    Z4 = (X4 / MX2)**2
    Z5 = X5

    X = np.stack([Z1,Z2,Z3,Z4,Z5],axis=1)
    for i in range(5,args.x_dim):
        x = np.random.normal(i/100, 1, (nobs,1))
        if i < 20:
            x = np.exp(x)
        X = np.concatenate((X, x), axis=1)

    muA = 5 * np.abs(X1) + 6 * np.abs(X2) + 3 * np.abs(X5) + np.abs(X4)

    A = np.random.noncentral_chisquare(3, muA, nobs)

    A = A.reshape(-1,1)
    for i in range(1, args.t_dim):
        mut = 0.1*X[:, i]**2

        t = (mut + np.random.normal(0, 0.5, nobs)).reshape(-1,1)
        A = np.concatenate((A, t), axis=1)
    A *= 0.1
    Cnum = ((MX1+3)**2+1) + 2*((MX2-25)**2+1)
    A_mean =  np.mean(A, axis=1).squeeze()
    Y = - 1.5 * A_mean - 15 + (X1+3)**2 + 2 * (X2-25)**2 + X3 - Cnum + np.random.normal(scale=1,size=nobs)
    Y /= 50

    for i in range(5,args.x_dim):
        if i % 2 == 0:
            Y += X[:,i].squeeze()
    
    Y = Y.reshape(-1,1)
    Y /= 4

    data = np.concatenate((A,X,Y),axis=1)

    adrf_t = np.linspace(np.min(A), 50.0, num=args.n_adrf).reshape(-1,1)
    adrf_t = adrf_t.repeat(args.t_dim, axis=1)
    
    adrf_y = ADRF(args, adrf_t).reshape(-1,1)
    adrf = np.concatenate((adrf_t, adrf_y), axis=1)

    idrf_t = np.linspace(np.min(A), 50.0, num=args.n_idrf).reshape(-1,1)
    idrf_x = X[args.n_train: args.n_train+args.n_idrf,:]
    Cnum = ((MX1+3)**2+1) + 2*((MX2-25)**2+1)
    idrf_y = - 1.5 * idrf_t.squeeze() - 15 + (X1[args.n_train: args.n_train+args.n_idrf]+3)**2 + 2 * (X2[args.n_train: args.n_train+args.n_idrf]-25)**2 + X3[args.n_train: args.n_train+args.n_idrf] - Cnum + np.random.normal(scale=1,size=args.n_idrf)
    idrf_y = (idrf_y/50.0).reshape(-1,1)
    idrf_t = idrf_t.repeat(args.t_dim, axis=1)

    idrf = np.concatenate((idrf_t, idrf_x, idrf_y), axis=1)
    if args.model == 'vcnet' or args.model == 'drnet':
    
        for i in range(data.shape[1]-1):
            data[:, i] = (data[:, i] - data[:, i].min()) / (data[:, i].max() - data[:, i].min())
        for i in range(adrf.shape[1]-1):
            adrf[:, i] = (adrf[:, i] - adrf[:, i].min()) / (adrf[:, i].max() - adrf[:, i].min())
        for i in range(idrf.shape[1]-1):
            idrf[:, i] = (idrf[:, i] - idrf[:, i].min()) / (idrf[:, i].max() - idrf[:, i].min())

    train_data = data[:args.n_train,:]
    test_data = data[args.n_train: args.n_train+args.n_val,:]
    val_data = data[args.n_train+args.n_val:,:]



    
    
    return train_data, val_data, test_data, adrf, idrf



if __name__ == '__main__':
    A,X,Y = simulate()
    plt.hist(A)
    plt.show()