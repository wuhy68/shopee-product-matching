import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
sys.path.append('../classification/timm')

import time
import random
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.backends import cudnn

from torch.cuda.amp import autocast as autocast

from dataset import ShopeeDataset, get_transforms
from util import global_average_precision_score, replace_activations, GradualWarmupSchedulerV2, ShopeeScheduler
from loss import Ranger
from models import ArcFaceLossAdaptiveMargin, Mish, Model_Shopee

from warnings import filterwarnings
filterwarnings("ignore")

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--net-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=4e-5)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--start-from-epoch', type=int, default=1)
    parser.add_argument('--stop-at-epoch', type=int, default=999)
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./models')
    parser.add_argument('--log-dir', type=str, default='./models')
    parser.add_argument('--fold', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

def gather_tensor(tensor):
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensor
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)

def getMetric(col):
    def f1score(row):
        n = len( np.intersect1d(row.target,row[col]) )
        return 2*n / (len(row.target)+len(row[col]))
    return f1score

def combine_for_cv(row):
    x = np.concatenate([row.preds])
    return np.unique(x)

def train_epoch(model, loader, optimizer, criterion, local_rank, epoch, n_epochs, fold_id, model_dir, kernel_type):

    model.train()
    losses = []
    scaler = torch.cuda.amp.GradScaler()

    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.cuda(), target.cuda()
        # with torch.no_grad():
        #     _, logits_t = teacher_model(data)

        optimizer.zero_grad()

        # torch.amp
        with autocast():
            feat, logits_m = model(data)
            loss = criterion(logits_m, target)
            # feat, logits_m = model(data)
            # loss = criterion(logits_t, logits_m, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss = reduce_mean(loss, torch.distributed.get_world_size())
        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        if ((batch_idx + 1) % 50 == 0 or batch_idx == len(loader) - 1 or batch_idx == 0) and local_rank == 0:
            print('Epoch: {}/{} Step: {}/{} Loss: {:.4f} Aug_loss: {:.4f} Smooth_loss: {:.4f}'.format(epoch, n_epochs, batch_idx + 1, len(loader), loss.item(), np.mean(losses), smooth_loss))
            with open(f'{model_dir}/{kernel_type}_fold{fold_id}_train.txt', 'a') as txt:
                print('Epoch: {}/{} Step: {}/{} Loss: {:.4f} Aug_loss: {:.4f} Smooth_loss: {:.4f}'.format(epoch, n_epochs, batch_idx + 1, len(loader), loss.item(), np.mean(losses), smooth_loss), file=txt)

    return np.mean(losses)

def val_epoch(model, valid_loader, criterion, margins, out_dim, get_output=False):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []
    LOGITS_M = []
    feats = []

    with torch.no_grad():
        for (data, target) in valid_loader:
            data, target = data.cuda(), target.cuda()

            # feat, logits_m3, logits_m = model(data)
            feat, logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            LOGITS_M.append(logits_m.detach().cpu())
            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            # loss = ArcFaceLossAdaptiveMargin(margins=margins, s=80)(logits_m, target, out_dim)
            loss = criterion(logits_m, target)
            loss = reduce_mean(loss, torch.distributed.get_world_size())
            val_loss.append(loss.item())
            feats.append(feat.detach().cpu())

        val_loss = np.mean(val_loss)

        PRODS_M = gather_tensor(torch.cat(PRODS_M).cuda()).cpu().numpy()
        PREDS_M = gather_tensor(torch.cat(PREDS_M).cuda()).cpu().numpy()
        TARGETS = gather_tensor(torch.cat(TARGETS).cuda()).cpu().numpy()

    if get_output:
        return LOGITS_M
    else:
        acc_m = (PREDS_M == TARGETS).mean() * 100.
        y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
        y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
        gap_m = global_average_precision_score(y_true, y_pred_m)

        return val_loss, acc_m, gap_m

def main():

    # get dataframe
    df = pd.read_csv('./data/shopee-product-matching/my_train_folds.csv')

    out_dim = df.label_group.nunique()
    if args.local_rank == 0:
        print(f'out_dim: {out_dim}')

    # get adaptive margin
    tmp = np.sqrt(1 / np.sqrt(df['label_group'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # get augmentations
    transforms_train, transforms_val = get_transforms(args.image_size)

    # get train and valid dataset
    df_train = df[df['fold'] != args.fold]
    df_valid = df[df['fold'] == args.fold]

    dataset_train = ShopeeDataset(df_train, './data/shopee-product-matching/train_images/', 'train', transform=transforms_train)
    dataset_valid = ShopeeDataset(df_valid, './data/shopee-product-matching/train_images/','train', transform=transforms_val)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    model = ModelClass(args.net_type, out_dim=out_dim)
    # teacher_model = ModelClass('resnet152d', out_dim=out_dim)
    # teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model).cuda()
    # teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    #
    # # # load pretrained
    # load_from = './models/r152d_512_bs8_f0_10ep/r152d_512_bs8_f0_10ep_lr4e-5_fold0.pth'
    # if load_from is not None:
    #     checkpoint = torch.load(load_from,  map_location='cpu')
    #     state_dict = checkpoint['model_state_dict']
    #     state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    #
    #     teacher_model.load_state_dict(state_dict, strict=False)
    #
    #     del checkpoint, state_dict
    #     torch.cuda.empty_cache()
    #     import gc
    #     gc.collect()

    # loss func
    # def criterion(logits_t, logits_m, target):
    #     loss_t = ArcFaceLossAdaptiveMargin(margins=margins, s=80)(logits_t, target, out_dim)
    #     loss_m = ArcFaceLossAdaptiveMargin(margins=margins, s=80)(logits_m, target, out_dim)
    #     return loss_m + 0.1 * loss_t

    def criterion(logits_m, target):
        loss_m = ArcFaceLossAdaptiveMargin(margins=margins, s=80)(logits_m, target, out_dim)
        return loss_m

    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    optimizer = Ranger(model.parameters(), lr=args.init_lr)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    existing_layer = torch.nn.SiLU
    new_layer = Mish()
    model = replace_activations(model, existing_layer, new_layer)
    # lr scheduler
    # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs-1)
    # scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    scheduler = ShopeeScheduler(optimizer, lr_start=args.init_lr)
    # train & valid loop
    gap_m_max = 0.
    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}.pth')
    for epoch in range(args.start_from_epoch, args.n_epochs+1):

        scheduler.step(epoch - 1)

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        train_sampler.set_epoch(epoch)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=train_sampler is None, sampler=train_sampler, drop_last=True)        

        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.local_rank, epoch, args.n_epochs, args.fold, args.log_dir, args.kernel_type)
        val_loss, acc_m, gap_m = val_epoch(model, valid_loader, criterion, margins, out_dim)

        if args.local_rank == 0:
            content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
            print(content)
            with open(os.path.join(args.log_dir, f'{args.kernel_type}.txt'), 'a') as appender:
                appender.write(content + '\n')

            if gap_m_max < gap_m:
                print('gap_m_max ({:.6f} --> {:.6f}). Saving model ...'.format(gap_m_max, gap_m))
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, model_file)
                gap_m_max = gap_m

        if epoch == args.stop_at_epoch:
            print(time.ctime(), 'Training Finished!')
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}_final.pth'))

if __name__ == '__main__':

    args = parse_args()
    torch.distributed.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    ModelClass = Model_Shopee

    set_seed(42)

    main()
