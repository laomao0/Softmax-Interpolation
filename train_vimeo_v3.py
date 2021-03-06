import os
from tkinter.tix import Tree
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import Vimeo90k
from torch.utils.data import DataLoader
# from SoftSplatModel import SoftSplatBaseline
# from SoftSplatModel_v2 import SoftSplatBaseline_v2
from SoftSplatModel_v3 import SoftSplatBaseline_v3
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
from validation import Vimeo_PSNR as validate
from lr_scheduler import ReduceLROnPlateau

Debug = False

def train():
    parser = ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='SoftSplatSamllBaseline_v3', help='experiment name')
    parser.add_argument('--n_epochs', type=int, default=80, help='number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='batch size for validation')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--loss_type', choices=['L1', 'MSE', 'Lap'], default='L1', help='loss function to use')
    parser.add_argument('--resume', type=int, default=0, help='epoch # to start / resume from.')
    parser.add_argument('--data_path', type=str, default='/DATA/wenbobao_data/vimeo_triplet', help='path to dataset (Vimeo90k)')
    parser.add_argument('--save_dir', type=str, default='./ckpt', help='path to tensorboard log')
    parser.add_argument('--log_dir', type=str, default='./logs', help='path to tensorboard log')
    parser.add_argument('--val_dir', type=str, default='./valid', help='path to save validation results')
    parser.add_argument('--valid_batch_size', type=int, default=4, help='batch size for validation')
    args = parser.parse_args()
    print(args)

    # paths
    save_path = f'{args.save_dir}/{args.exp_name}.pth'
    logs = f'{args.log_dir}/{args.exp_name}'
    valid_path = f'{args.val_dir}/{args.exp_name}'

    # model
    # model = SoftSplatBaseline()
    model = SoftSplatBaseline_v3()
    model = nn.DataParallel(model).cuda()

    # optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # dataset
    train_data = Vimeo90k(args.data_path)
    test_data = Vimeo90k(args.data_path, is_train=False)
    ipe = len(train_data) // args.batch_size
    print('iterations per epoch:', ipe)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(test_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=0)

    # loss
    best = 0
    loss_fn = None
    if args.loss_type == 'L1':
        loss_fn = nn.L1Loss()
    elif args.loss_type == 'MSE':
        loss_fn = nn.MSELoss()
    elif args.loss_type == 'Lap':
        from laplacian import LapLoss 
        loss_fn = LapLoss()

    if not args.resume == 0:  # if resume training
        print('loading checkpoints...')
        ckpt = torch.load(save_path)
        model.module.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        optimizer.param_groups[0]['lr'] = args.lr
        del ckpt
        print('load complete!')
    else:
        if os.path.exists(logs):
            shutil.rmtree(logs)
        if os.path.exists(valid_path):
            shutil.rmtree(valid_path)

    # recording & tracking
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    writer = SummaryWriter(logs)
    prev_best = None

    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.2, patience=5, verbose=True)

    ikk = 0
    for kk in optimizer.param_groups:
        if kk['lr'] > 0:
            ikk = kk
            break

    print('start training.')
    for epoch in range(args.resume, args.n_epochs):
        epoch_train_loss = 0
        model.train()
        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):

            if Debug == True:
                if i >= 10:
                    break

            input_frames, target_frame, target_t, _ = data
            input_frames = input_frames.cuda().float()
            target_frame = target_frame.cuda().float()
            target_t = target_t.cuda().float()

            # forwarding
            pred_frt = model(input_frames, target_t)
            total_loss = loss_fn(pred_frt, target_frame).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_train_loss += total_loss.item()

        epoch_train_loss /= ipe
        torch.cuda.empty_cache()
        valid_psnr, valid_loss, cur_val_path = validate(model, valid_loader, valid_path, epoch, Debug)
        torch.cuda.empty_cache()
        writer.add_scalar('PSNR', valid_psnr, epoch)
        writer.add_scalar(f'{args.loss_type}/Train', epoch_train_loss, epoch)
        writer.add_scalar(f'{args.loss_type}/Valid', valid_loss, epoch)
        writer.add_scalar(f'{args.loss_type}/LR', float(ikk['lr']), epoch)


        if valid_psnr > best:
            best = valid_psnr
            ckpt = {'opt': optimizer.state_dict(), 'model': model.module.state_dict()}
            torch.save(ckpt, save_path)
            # remove previous best validation.
            if prev_best is not None:
                shutil.rmtree(prev_best)
            prev_best = cur_val_path
        else:
            if prev_best is not None:
                shutil.rmtree(cur_val_path)

        scheduler.step(valid_loss)

    print('end of training.')
    print('final validation.')
    torch.cuda.empty_cache()
    valid_psnr, valid_loss, cur_val_path = validate(model, valid_loader, valid_path, args.n_epochs, Debug)
    torch.cuda.empty_cache()
    writer.add_scalar('PSNR', valid_psnr, args.n_epochs)
    if valid_psnr > best:
        best = valid_psnr
        ckpt = {'opt': optimizer.state_dict(), 'model': model.module.state_dict()}
        torch.save(ckpt, save_path)
        # remove previous best validation.
        if prev_best is not None:
            shutil.rmtree(prev_best)
    else:
        if prev_best is not None:
            shutil.rmtree(cur_val_path)
    print(f'Final model PSNR: {best.item()}')

    


if __name__ == '__main__':
    train()
