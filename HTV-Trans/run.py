from trainer import Trainer
from newtester import Tester
import numpy as np
import os
import argparse
import torch
from models import StackedVAGT
from dataloader import data_provider
import random
import time

# the same as NS former
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

parser = argparse.ArgumentParser()
# GPU
parser.add_argument('--gpu_id', type=int, default=0)
# Dataset options
parser.add_argument('--dataset', default='custom', type=str, help='data type')
parser.add_argument('--data_path', type=str, default='exchange_rate.csv', help='data file name')
parser.add_argument('--root_path', type=str, default='./data/exchange_rate/',
                    help='root path of the data file')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length no use')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--x_dim', type=int, default=8, help='input dim')
parser.add_argument('--win_len', type=int, default=48, help='not use')
# Model options for HTV_trans
parser.add_argument('--z_dim', type=int, default=25, help='not use')
parser.add_argument('--h_dim', type=int, default=128)
parser.add_argument('--n_head', type=int, default=8, help='8')
parser.add_argument('--layer_xz', type=int, default=2)
parser.add_argument('--z_layers', type=int, default=1, help='transformer layers')
parser.add_argument('--q_len', type=int, default=1, help='for conv1D padding in Transformer')
parser.add_argument('--embd_h', type=int, default=48, help='embedding dim')
parser.add_argument('--embd_s', type=int, default=16, help='not use')
parser.add_argument('--vocab_len', type=int, default=512, help='256')
parser.add_argument('--decoder_type', type=str, default='fcn', help='choose decoder:[trans,fcn]')
parser.add_argument('--mode_type', type=str, default='train', help='train or test')
# Training options for HTV_trans
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--anneal_rate', type=float, default=0.05)
parser.add_argument('--coeff', type=float, default=0.01)
parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
# Save options
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--checkpoints_interval', type=int, default=1)
parser.add_argument('--checkpoints_path', type=str, default='model/debug')
parser.add_argument('--checkpoints_file', type=str, default='')
parser.add_argument('--log_path', type=str, default='log_path/debug')
parser.add_argument('--log_file', type=str, default='')

args = parser.parse_args()

def _get_data(args, flag):
    data_set, data_loader = data_provider(args, flag)
    return data_set, data_loader

# Set up GPU
if torch.cuda.is_available() and args.gpu_id >= 0:
    device = torch.device('cuda:%d' % args.gpu_id)
else:
    device = torch.device('cpu')

# For config checking
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)

if not os.path.exists(args.checkpoints_path):
    os.makedirs(args.checkpoints_path)

# TODO Saving path names, for updating later...
if args.checkpoints_file == '':
    args.checkpoints_file = 'x_dim-{}_z_dim-{}_h_dim-{}_layer_xz-{}_z_layers-{}_embd_h-{}_n_head-{}_' \
                            'pred_len-{}_q_len-{}_vocab_len-{}'.format(args.x_dim, args.z_dim, args.h_dim,
                                                                      args.layer_xz, args.z_layers, args.embd_h,
                                                                      args.n_head, args.pred_len, args.q_len,
                                                                      args.vocab_len)
if args.log_file == '':
    args.log_file = 'x_dim-{}_z_dim-{}_h_dim-{}_layer_xz-{}_z_layers-{}_embd_h-{}_n_head-{}_pred_len-{}_' \
                    'q_len-{}_vocab_len-{}'.format(args.x_dim, args.z_dim, args.h_dim, args.layer_xz,
                                                   args.z_layers, args.embd_h, args.n_head, args.pred_len,
                                                   args.q_len, args.vocab_len)


_, train_loader = _get_data(args, flag='train')
_, val_loader = _get_data(args, flag='val')
_, test_loader = _get_data(args, flag='test')


# For models init
stackedvagt = StackedVAGT(layer_xz=args.layer_xz, z_layers=args.z_layers, n_head=args.n_head, x_dim=args.x_dim,
                          z_dim=args.z_dim, h_dim=args.h_dim, embd_h=args.embd_h, embd_s=args.embd_s,
                          beta=args.beta, q_len=args.q_len, vocab_len=args.vocab_len, win_len=args.seq_len,
                          horizon=args.pred_len, label_len=args.label_len,
                          dropout=args.dropout, anneal_rate=args.anneal_rate, max_beta=args.max_beta,
                          device=device, decoder_type=args.decoder_type).to(device)
names = []
for name, parameters in stackedvagt.named_parameters():
    names.append(name)

trainer = Trainer(stackedvagt, train_loader, val_loader, log_path=args.log_path, epochs=args.epochs,
                  log_file=args.log_file, batch_size=args.batch_size, learning_rate=args.learning_rate,
                  coeff=args.coeff, patience=args.patience,
                  checkpoints=os.path.join(args.checkpoints_path, args.checkpoints_file),
                  checkpoints_interval=args.checkpoints_interval, device=device)


total = sum([param.nelement() for param in stackedvagt.parameters()])
total += sum(p.numel() for p in stackedvagt.buffers())
print("Number of parameters: %.2fM" % (total/(1024*1024)))

# train model
if(args.mode_type == 'train'):
    trainer.load_checkpoint(args.start_epoch, trainer.checkpoints)
    trainer.train_model()


tester = Tester(stackedvagt, test_loader, log_path=args.log_path,
                log_file=args.log_file, learning_rate=args.learning_rate, device=device,
                checkpoints=os.path.join(args.checkpoints_path, args.checkpoints_file),
                nsamples=None, sample_path=None)


if(args.mode_type == 'train' or args.mode_type == 'test'):
    for i in range(0, round(args.epochs) * args.checkpoints_interval, args.checkpoints_interval):
        if(tester.load_checkpoint(args.epochs - i)):
            tester.load_checkpoint(args.epochs - i)
            print('NOW Epoch is {:.0f}'.format(args.epochs - i))
            # inference test
            since = time.time()
            tester.model_test_v2()
            time_elapsed = time.time() - since
            print('Complete in {:.5f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))
            torch.cuda.empty_cache()
            break


