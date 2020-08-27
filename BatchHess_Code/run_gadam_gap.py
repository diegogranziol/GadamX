import argparse
import os
import sys
import time
import tabulate
import numpy as np
import torch

from curvature import data, models, losses, utils
from optimizers import Padam

parser = argparse.ArgumentParser(description='SGD/SWA training')
#parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true',
                    help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to averaging from (default: None)')

parser.add_argument('--init_epochs', type=int, default=0, metavar='N', help='number of epochs for pretraining (default: 10)')
parser.add_argument('--save_freq', type=int, default=10, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--decoupled_wd', action='store_true', help="Enable to use AdamW - decoupled weight decay")
parser.add_argument("--normalized_wd", action='store_true',
                    help='Whether to use normalised wd. WD = WD_norm \sqrt(\frac{b}{BT})')
parser.add_argument("--partial", type=float, default=0.125, help='Padam partial parameter')

parser.add_argument('--swa_epochs', type=int, default=100, metavar='N', help='number of epochs for swa (default: 100)')
parser.add_argument('--swa_gap', type=int, default=1, metavar='N', help='swa averaging gap (default: 1)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# print('Preparing directory %s' % args.dir)
# os.makedirs(args.dir, exist_ok=True)
# with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
#     f.write(' '.join(sys.argv))
#     f.write('\n')

args.dir = args.ckpt[:-19]
print('directory is '+args.dir)

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

print('SWA training')
swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swa_model.to(args.device)

criterion = losses.cross_entropy

if args.normalized_wd:
    weight_decay = args.wd * np.sqrt(args.batch_size / (args.epochs * args.batch_size * len(loaders['train'])))
else:
    weight_decay = args.wd

optimizer = Padam(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
    amsgrad=False,
    decoupled_wd=args.decoupled_wd,
    partial=args.partial
)

print('Loading %s' % args.ckpt)
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['state_dict'])

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'mem_usage']


for epoch in range(args.init_epochs):
    time_ep = time.time()
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.init_epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

    values = [args.init_epochs + epoch + 1, args.lr, train_res['loss'], train_res['accuracy'], test_res['loss'],
              test_res['accuracy'],
              test_res['top5_accuracy'], time_ep, memory_usage]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.init_epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.init_epochs,
        name='init',
        epoch=args.init_epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict()
    )

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'te_top5_acc', 'time', 'mem_usage']
columns = columns[:-2] + ['swa_tr_loss', 'swa_tr_acc', 'swa_te_loss', 'swa_te_acc', 'swa_te_top5_acc'] + columns[-2:]
swa_res = {'loss': None, 'accuracy': None, 'top5_accuracy': None}


t = 0
n_swa = 0
for epoch in range(0, args.swa_epochs):
    time_ep = time.time()

    loss_sum = 0.0
    correct = 0.0
    for input, target in loaders['train']:
        if args.device.type == 'cuda':
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output, stats = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)

        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

        if t % args.swa_gap == 0:
            for param, swa_param in zip(model.parameters(), swa_model.parameters()):
                swa_param.data += (param.data - swa_param.data) / (n_swa + 1)
            n_swa += 1

        t += 1

    loss_sum /= len(loaders['train'].dataset)
    correct /= len(loaders['train'].dataset)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.swa_epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
        utils.bn_update(loaders['train'], swa_model)
        train_swa_res = utils.eval(loaders['train'], swa_model, criterion)
        swa_res = utils.eval(loaders['test'], swa_model, criterion)
        time_ep = time.time() - time_ep
        memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

        values = [args.init_epochs + epoch + 1, args.lr, train_res['loss'], train_res['accuracy'], test_res['loss'],
                  test_res['accuracy'],
                  test_res['top5_accuracy'], time_ep, memory_usage]
        values = values[:-2] + [train_swa_res['loss'], train_swa_res['accuracy'],
                                swa_res['loss'], swa_res['accuracy'], swa_res['top5_accuracy']] + values[-2:]

        np.savez(
            args.dir + 'Padam-nogap-stats-' + str(args.init_epochs) + str(epoch+1),
            train_loss=train_res['loss'],
            time_ep=time_ep,
            memory_usage=memory_usage,
            train_accuracy=train_res['accuracy'],
            train_top5_accuracy=train_res['top5_accuracy'],
            test_loss=test_res['loss'],
            test_accuracy=test_res['accuracy'],
            test_top5_accuracy=test_res['top5_accuracy'],
            swag_loss=swa_res['loss'],
            swag_train_loss=train_swa_res['loss'],
            swag_train_acc=train_swa_res['accuracy'],
            swag_accuracy=swa_res['accuracy'],
            swag_top5_accuracy=swa_res['top5_accuracy']
        )

    else:
        test_res = {'loss': None, 'accuracy': None}
        swa_res = {'loss': None, 'accuracy': None}

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)







    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            args.init_epochs + epoch + 1,
            epoch=args.init_epochs + epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict()
        )

        utils.save_checkpoint(
            args.dir,
            args.init_epochs + epoch + 1,
            name='swa',
            epoch=args.init_epochs + epoch + 1,
            state_dict=swa_model.state_dict(),
        )

# utils.save_checkpoint(
#     args.dir,
#     args.init_epochs + args.swa_epochs,
#     epoch=args.init_epochs + args.swa_epochs,
#     state_dict=model.state_dict(),
#     optimizer=optimizer.state_dict()
# )
#
# utils.save_checkpoint(
#     args.dir,
#     args.init_epochs + args.swa_epochs,
#     name='swa',
#     epoch=args.init_epochs + args.swa_epochs,
#     state_dict=swa_model.state_dict(),
# )