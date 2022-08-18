from models.causal_forest import CausalForest
from models.bart import BART
from models.gps import GPS
from models.cbgps import CBGPS
from models.eb import EB
from models.dcows import DCOWS
from models.drnet import Drnet
from models.vcnet import Vcnet
from torch.optim import Adam

def train_model(args, train_data, val_data):
    if args.model == 'cf':
        stg1 = CausalForest(args.t_dim)
        model = stg1.train_model(train_data)

    if args.model == 'bart':
        stg1 = BART(args.t_dim)
        model = stg1.train_model(args, train_data)

    if args.model == 'gps':
        stg1 = GPS(args.t_dim)
        model = stg1.train_model(train_data)

    if args.model == 'cbgps':
        stg1 = CBGPS(args.t_dim)
        model = stg1.train_model(train_data)
        
    if args.model == 'eb':
        stg1 = EB(args.t_dim)
        model = stg1.train_model(train_data)

    if args.model == 'dcows':
        stg1 = DCOWS(args.t_dim)
        model = stg1.train_model(train_data)
    
    if args.model == 'drnet':

        cfg_density = [(args.x_dim+args.t_dim-1, 50, 1, 'relu'), (50, 50, 1, 'relu')]
        num_grid = 10
        cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
        isenhance = 1
        stg1 = Drnet(cfg_density, num_grid, cfg, isenhance)
        stg1._initialize_weights()
        # opt = SGD(model.parameters(), lr=5e-2)
        opt = Adam(stg1.parameters(), lr=args.lr)
        model = stg1.train_model(opt, train_data)
    
    if args.model == 'vcnet':
        cfg_density = [(args.x_dim+args.t_dim-1,50,1,'relu'), (50,50,1,'relu')]
        num_grid = 10
        cfg = [(50,50,1,'relu'), (50,1,1,'id')]
        degree = 2
        knots = args.knots
        stg1 = Vcnet(cfg_density, num_grid, cfg, degree, knots)
        stg1._initialize_weights()
        # opt = SGD(model.parameters(), lr=0.001)
        opt = Adam(stg1.parameters(), lr=args.lr)
        model = stg1.train_model(opt, train_data)
    return model