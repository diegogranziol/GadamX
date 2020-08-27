import argparse
import os
import sys
import time
import tabulate

import torch

from curvature import data, models, losses, utils
from curvature.methods.swag import SWAG
import numpy as np
print('numpy imported')

from gpytorch.utils.lanczos import lanczos_tridiag

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--bn_train_mode_off', action='store_true')

parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=1, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--no_schedule', action='store_true', help='store schedule')

parser.add_argument('--swag', action='store_true')
parser.add_argument('--swag_subspace', choices=['covariance', 'pca', 'freq_dir'], default='pca')
parser.add_argument('--swag_rank', type=int, default=20, help='SWAG covariance rank')
parser.add_argument('--swag_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swag_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swag_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--swag_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

#added DIEGO Lanczos arguments
parser.add_argument('--iters', type=int, default=20, metavar='N', help='number of lanczos steps (default: 20)')
parser.add_argument('--spectrum_path', type=str, default=None, metavar='PATH',
                    help='path to save spectrum (default: None)')
parser.add_argument('--basis_path', type=str, default=None, metavar='PATH',
                    help='path to save Lanczos vectors  (default: None)')
parser.add_argument('--warmstart', type=int, default=-1, metavar='N', help='number of epoch warm starts (default: 10)')
parser.add_argument('--epochfreq', type=int, default=20, metavar='N', help='number of epochs at learned learning rate (default: 10)')
parser.add_argument('--curvaturebatchsize', type=int, default=1024, metavar='N', help='number of epochs at learned learning rate (default: 10)')



args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.dir = args.dir  + args.dataset + '/' + args.model + '/SSGDM'
# thing = ""
# if args.no_schedule:
#     thing += "_flat"
# if args.step_schedule:
#     thing += "_step"
# if args.swag:
#     args.dir += "SWA"
#     thing = "_swalr="+str(args.swag_lr)+'_swastart='+str(args.swag_start)
args.dir += '/seed='+str(args.seed)+'_epoch_freq='+str(args.epochfreq)+'_curvaturesize='+str(args.curvaturebatchsize)+'_warmstart='+str(args.warmstart)+ '_wd=' + str(
    args.wd) +'_numepochs=' + str(args.epochs) + '/'


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

print('Preparing model')
print(*model_cfg.args, dict(**model_cfg.kwargs))
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

if args.swag:
    print('SWAG training')
    swag_model = SWAG(model_cfg.base,
                      subspace_type=args.swag_subspace, subspace_kwargs={'max_rank': args.swag_rank},
                      *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swag_model.to(args.device)
else:
    print('SGD training')


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

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.swag and args.swag_resume is not None:
    checkpoint = torch.load(args.swag_resume)

    swag_model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'mem_usage']
if args.swag:
    columns = columns[:-2] + ['swa_te_loss', 'swa_te_acc'] + columns[-2:]
    swag_res = {'loss': None, 'accuracy': None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    epoch=start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict()
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    lr = args.lr_init
    #ADD NEW CODE BY DIEGO
    eigvel = []
    gammi = []

    datasets, num_classes = data.datasets(
        args.dataset,
        args.data_path,
        transform_train=model_cfg.transform_test,
        transform_test=model_cfg.transform_test,
        use_validation=False,
        train_subset=args.curvaturebatchsize,
        train_subset_seed=5,
    )
    loaderbeast = torch.utils.data.DataLoader(
        datasets['train'],
        batch_size=args.curvaturebatchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    if epoch >= args.warmstart and epoch % args.epochfreq == 0:
        model.to(args.device)
        num_parametrs = sum([p.numel() for p in model.parameters()])
        class HessVecProduct(object):
            def __init__(self, loader, model, criterion):
                self.loader = loader
                self.model = model
                self.criterion = criterion
                self.iters = 0
                self.timestamp = time.time()

            def __call__(self, vector):
                start_time = time.time()
                output = utils.hess_vec(vector, self.loader, self.model, self.criterion, cuda=args.device.type == 'cuda',
                bn_train_mode = not args.bn_train_mode_off)
                time_diff = time.time() - start_time
                self.iters += 1
                print('Iter %d. Time: %.2f' % (self.iters, time_diff))
                return output.cpu().unsqueeze(1)


        productor = HessVecProduct(loaderbeast, model, criterion)

        Q, T = lanczos_tridiag(productor, args.iters, dtype=torch.float32, device='cpu',
                               matrix_shape=(num_parametrs, num_parametrs))

        eigvals, eigvects = T.eig(eigenvectors=True)
        gammas = eigvects[0, :] ** 2
        V = eigvects.t() @ Q.t()
        #eigvals = eigvals*float(num_parametrs / args.batch_size)

        order = np.argsort(np.abs(eigvals.numpy()[:, 0]))[::-1]
        table = [[eigvals[order[i], 0].item(), gammas[order[i]].item()] for i in range(order.size)]
        print(tabulate.tabulate(table, headers=['value', 'weight'], tablefmt='simple'))

        eigvals = np.sort(np.abs(eigvals))
        eigvals = np.sum(eigvals, axis=1)

        #losstats = utils.loss_stats(loaderbeast, model, criterion, cuda=True, bn_train_mode=True, verbose=False)
        #alpha = losstats['alpha'].item()
        #print(alpha)

        lr = 2./(np.amax(eigvals)+np.median(eigvals))
        print('this is the learning rate')
        print(lr)
        utils.adjust_learning_rate(optimizer, lr)

        # # Potential Edit to the learning rate
        # loss_stats = utils.loss_stats(stats_loader, model, criterion, cuda=True, bn_train_mode=False)
        # alpha = loss_stats['alpha'].item()

        # lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)

        eigvel.append(eigvals)
        gammi.append(np.array(gammas))

        lrlearned = lr

        np.savez(
            args.dir + 'lanczos-' + str(epoch),
            eigvals=eigvals,
            gammas=gammas,
            lr=lrlearned,
            mom=momlearned
        )

    if epoch >= args.warmstart:
        utils.adjust_learning_rate_and_momentum(optimizer, lrlearned, momlearned)
        lr = lrlearned
        momentum = momlearned

        if args.spectrum_path is not None:
            np.savez(
                args.dir+'-'+str(epoch),
                eigvals=eigvals,
                gammas=gammas
            )

        if args.basis_path is not None:
            torch.save(
                {
                    'w': torch.cat([param.detach().cpu().view(-1) for param in model.parameters()]),
                    'eigvals': eigvals,
                    'gammas': gammas,
                    'V': V,
                },
                args.dir+'-'+str(epoch),
            )

    #lr = schedule(epoch)
    #utils.adjust_learning_rate_and_momentum(optimizer, lr, momentum)

    #END ADD NEW CODE DIEGO

    # if not args.no_schedule:
    #     lr = schedule(epoch)
    #     utils.adjust_learning_rate(optimizer, lr)
    # else:
    #     lr = args.lr_init

    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
    
    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}

    if args.swag and (epoch + 1) > args.swag_start:
        # If the frequency of collecting swag models is less than once per epoch - otherwise the models have been
        # collected already in the train_epoch call.
        if args.swag_c_epochs >= 1 and (epoch + 1 - args.swag_start) % args.swag_c_epochs == 0:
            swag_model.collect_model(model)

        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.set_swa()
            utils.bn_update(loaders['train'], swag_model)
            train_res_swag = utils.eval(loaders['train'], swag_model, criterion, optimizer)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)

        else:
            swag_res = {'loss': None, 'accuracy': None, "top5_accuracy": None}
            train_res_swag = {'loss': None, 'accuracy': None}

    else:
        train_res_swag = {'loss': None, 'accuracy': None}

    np.savez(
        args.dir + 'stats-' + str(epoch),
        train_loss=train_res['loss'],
        train_accuracy=train_res['accuracy'],
        test_loss=test_res['loss'],
        test_accuracy=test_res['accuracy'],
    )


    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            epoch=epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )
        if args.swag and (epoch + 1) > args.swag_start:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag',
                epoch=epoch + 1,
                state_dict=swag_model.state_dict(),
            )
            np.savez(
                args.dir + 'stats-' + str(epoch),
                train_loss=train_res['loss'],
                train_accuracy=train_res['accuracy'],
                test_loss=test_res['loss'],
                test_accuracy=test_res['accuracy'],
                swag_loss=swag_res['loss'],
                swag_accuracy=swag_res['accuracy']
            )
    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3)

    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
              time_ep, memory_usage]
    if args.swag:
        values = values[:-2] + [swag_res['loss'], swag_res['accuracy']] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

    if np.isnan(train_res['loss']):
        break

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
