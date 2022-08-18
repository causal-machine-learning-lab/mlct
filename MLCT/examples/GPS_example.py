import sys
sys.path.append('/Users/zmin/Desktop/Myjob/survey-outline/CausalCT/CausalCT')

from argparse import ArgumentParser
import os
from models import *
from datasets.simulation import simulate
from train_model import train_model
from metrics.adrf import ADRF
from metrics.idrf import IDRF
import warnings

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    parser = ArgumentParser()

    # Set Hyperparameters
    parser.add_argument('--data_dir', type=str, default='dataset/simulation_low_d', help='dir of data')
    parser.add_argument('--save_dir', type=str, default='logs/5t_cf', help='dir to save result')
    parser.add_argument('--ihdp', type=str, default='/Users/zmin/Desktop/DCHCT_V4_2/data', help='dir to save result')

    parser.add_argument('--T_model', type=str, default=None,
                        help='model to use,[None, "gps", "bart", "cbgps", "npcbgps", "gbm", "eb", "dcows", "dcw", "vsr"]')
    parser.add_argument('--model', type=str, default='gps', help='model to use,["nn", "drnet", "vcnet"]')
    parser.add_argument('--n_epochs', type=int, default=3000, help='num of epochs to train')
    parser.add_argument('--n_exps', type=int, default=10, help="the number of experiments")
    parser.add_argument("--train_bs", default=1000, type=int, help='train batch size')
    parser.add_argument('--n_samples', type=int, default=2600, help="the number of generated samples")
    parser.add_argument('--n_train', type=int, default=2000, help="the number of samples for training")
    parser.add_argument('--n_adrf', type=int, default=300, help="the number of samples for training")
    parser.add_argument('--n_idrf', type=int, default=30, help="the number of samples for training")
    
    parser.add_argument('--t_dim', type=int, default=1, help="the dimension of treatments")
    parser.add_argument('--x_dim', type=int, default=5, help="the dimension of covariates")
    parser.add_argument('--t_bin', type=bool, default=False, help="treament is binary(True) or not(False)")
    parser.add_argument('--n_val', type=int, default=300, help="the number of samples for training")

    # print train info
    parser.add_argument('--verbose', type=int, default=500, help='print train info freq')
    parser.add_argument('--n_workers', type=int, default=0, help='num of workers')

    args = parser.parse_args()

    # generate train/va/test
    train_data, val_data, test_data, adrf, idrf = simulate(args, nobs=args.n_samples)

    # get the trained model
    model = train_model(args, train_data, val_data)
    
    adrf_mse = ADRF(args, model, test_data, adrf)

    idrf_mse = IDRF(args, model, idrf)
    print(adrf_mse, idrf_mse)
