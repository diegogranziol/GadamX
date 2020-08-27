import argparse
import os
import sys
import math
import time
import tabulate
import numpy as np
import torch

from curvature import data, models, losses, utils
from curvature import data_subset as data_subset
from curvature.methods.swag import SWAG
from curvature.methods.shrinkagegradopt import ShrinkageOpt

parser = argparse.ArgumentParser(description='SGD_noschedule/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True,
                    help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                    help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
                    help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR',
                    help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD_noschedule momentum (default: 0.9)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')

#parser.add_argument('--turn_off', type=int, default=150, help='SWAG covariance rank')

parser.add_argument('--swag', action='store_true')
parser.add_argument('--swag_subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--swag_rank', type=int, default=20, help='SWAG covariance rank')
parser.add_argument('--swag_start', type=float, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR',
                    help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--num_samples', type=int, default=None, metavar='N', help='number of data points to use (default: the whole dataset)')
parser.add_argument('--subsample_seed', type=int, default=None, metavar='N', help='random seed for dataset subsamling (default: None')
parser.add_argument('--wd_freq', type=int, default=20, metavar='N', help='wd update frequency (default: 5)')
parser.add_argument('--stats_batch', type=int, default=128, metavar='B', help='batch size used to compute the statistics (default: 1)')
parser.add_argument('--wd_mode_off', action='store_true', help='turn off wd mode of the regularizer')
parser.add_argument('--clip_alpha', type=float, default=0.001, metavar='N', help='clip alpha (default: 0.01)')
parser.add_argument('--curvature_matrix', type=str, default='hessian', metavar='curvature matrix', help='curvature matrix to compute loss stats of')
parser.add_argument('--epochend', type=int, default=None, metavar='D', help='Epoch When to turn weight decay to 0')
parser.add_argument('--num_curv_samples', type=int, default=None, metavar='N', help='number of data points to use for curvature (default: the whole dataset)')
parser.add_argument('--bn_train_mode', action='store_true')



parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()
if args.num_curv_samples:
    extra = '_num_curv_samples='+str(args.num_curv_samples)
else:
    extra = ''
if args.swag:
    swagchat = '_swastart=' +str(args.swag_start) + '_slr=' +str(args.swag_lr)
else:
    swagchat = ''


args.dir = args.dir + args.dataset + '/' + args.model + '/Shrinkage_grad_SGD' + '/seed='+str(args.seed)+'_lr='+str(args.lr_init) + swagchat + '_wdfreq='+str(args.wd_freq)+'_matrix='+ args.curvature_matrix + '_wdend=' +str(args.epochend) + extra + '/'


args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
)

datasets, num_classes = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
    train_subset=args.num_samples,
    train_subset_seed=args.subsample_seed,
)

smalldatasets, num_classes = data.datasets(
    args.dataset,
    args.data_path,
    transform_train=model_cfg.transform_test,
    transform_test=model_cfg.transform_test,
    use_validation=not args.use_test,
    train_subset=args.num_curv_samples,
    train_subset_seed=1,
)

if args.num_curv_samples:
    stats_loader = torch.utils.data.DataLoader(
        smalldatasets['train'],
        batch_size=args.stats_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
else:
    stats_loader = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=args.stats_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

if args.swag:
    print('SWAG training')
    swag_model = SWAG(model_cfg.base,
                      subspace_type=args.swag_subspace,
                      subspace_kwargs={'max_rank': args.swag_rank},
                      *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swag_model.to(args.device)
else:
    print('SGD_noschedule training')


def schedule(epoch):
    t = epoch / (args.swag_start if args.swag else args.epochs)
    lr_ratio = args.swag_lr / args.lr_init if args.swag else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


criterion = losses.cross_entropy

optimizer = ShrinkageOpt(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    wd_mode=not args.wd_mode_off,
    clip_alpha=args.clip_alpha
)


start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
columns = ['ep', 'wd', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'te_top5_acc','alpha', 'gradmean', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_te_acc', 'swa_te_top5_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    epoch=start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)

alpha_ma = 1.0

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)
    norm = math.sqrt(sum([torch.sum(param ** 2).item() for param in model.parameters()]))

    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)

    utils.bn_update(loaders['train'], model)
    test_res = utils.eval(loaders['test'], model, criterion)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

    if args.swag and (epoch + 1) >= args.swag_start and (
            epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            train_res_swag = utils.train_epoch(loaders['train'], swag_model, criterion, optimizer)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)
        else:
            swag_res = {'loss': None, 'accuracy': None, "top5_accuracy": None}
            train_res_swag = {'loss': None, 'accuracy': None}

    if args.epochend is not None and epoch > args.epochend:
        print('no weight decay')
        wd = 0
        alpha, gradmean = None, None
    else:
        if epoch % args.wd_freq == 0:
            # print('learning weight decay')
            num_batches = len(stats_loader)
            loss_stats = utils.loss_stats_grad_complex(stats_loader, model, criterion, cuda=True,
                                                       bn_train_mode=args.bn_train_mode)
            #hess_var = loss_stats['hess_var'].item()
            grad_var = loss_stats['grad_var'].item()
            #delta = loss_stats['delta'].item()
            delta = loss_stats['delta'].item()
            gamma = loss_stats['gamma'].item()

            alpha = loss_stats['alpha'].item()
            #alpha = 1.0 - args.stats_batch * grad_var / len(stats_loader.dataset) / delta_g

            alpha_ma = 0.5 * alpha_ma + (1.0 - 0.5) * alpha
            gradmean = loss_stats['gmean'].item()
            #gradmean = max(0.0, loss_stats['gmean'].item())
            wd = (1.0 - alpha_ma) / alpha_ma * gradmean
            for param_group in optimizer.param_groups:
                param_group['alpha'] = alpha
                param_group['gradmean'] = gradmean
        else:
            alpha, gradmean, wd = None, None, None

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.swag:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag',
                epoch=epoch + 1,
                state_dict=swag_model.state_dict(),
            )

    time_ep = time.time() - time_ep

    values = [epoch + 1,  wd, lr, train_res['loss'], train_res['accuracy'], test_res['loss'],
              test_res['accuracy'], test_res['top5_accuracy'], alpha, gradmean, time_ep, memory_usage]

    if train_res['loss'] != train_res['loss']:
        break


    np.savez(
        args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
        train_accuracy=train_res['accuracy'],
        train_top5_accuracy = train_res['top5_accuracy'],
        test_loss=test_res['loss'],
        test_accuracy=test_res['accuracy'],
        test_top5_accuracy = test_res['top5_accuracy'],
        alpha=alpha,
        alpha_ma=alpha_ma,
        gradmean=gradmean,
        wd=wd,
        norm=norm,
        lr = lr,
        time_ep = time_ep
    )
    if args.swag:
        values = values[:-2] + [swag_res['loss'], swag_res['accuracy'], swag_res['top5_accuracy']] + values[-2:]
        np.savez(
            args.dir + 'stats-' + str(epoch),
            train_loss=train_res['loss'],
            train_accuracy=train_res['accuracy'],
            train_top5_accuracy=train_res['top5_accuracy'],
            test_loss=test_res['loss'],
            test_accuracy=test_res['accuracy'],
            test_top5_accuracy=test_res['top5_accuracy'],
            swag_loss= swag_res['loss'],
            swag_accuracy = swag_res['accuracy'],
            alpha=alpha,
            alpha_ma=alpha_ma,
            gradmean=gradmean,
            wd=wd,
            norm=norm,
            lr=lr,
            time_ep=time_ep
        )
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8g')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        epoch=args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )
    if args.swag:
        utils.save_checkpoint(
            args.dir,
            args.epochs,
            name='swag',
            epoch=args.epochs,
            state_dict=swag_model.state_dict(),
        )
