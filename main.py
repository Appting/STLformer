# Liuzhaoxi 2023/11/7 16:16
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=207)
parser.add_argument('--n_his', type=int, default=12)
parser.add_argument('--n_pred', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epoch', type=int,  default=100)
parser.add_argument('--save', type=int, default=10)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks=[[64,64]]
# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./data_loader/PeMS-M/METR-LA/', f'W_{n}.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))

# Calculate graph kernel
L = scaled_laplacian(W)
V,U=np.linalg.eig(L)
U_ini=U[:,:32]
print(U_ini.shape)
tf.add_to_collection(name='intial_spatial_embeddings', value=tf.cast(tf.constant(U_ini), tf.float32))
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = f'V_{n}.csv'
n_train, n_val, n_test = 34, 5, 5
PeMS = data_gen(pjoin('./data_loader/PeMS-M/METR-LA/', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS,args.batch_size, n_his, n_pred, args.inf_mode)
