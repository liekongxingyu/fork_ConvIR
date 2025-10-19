import os
import torch
from data import train_dataloader
from utils import Adder, Timer
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
import torch.nn as nn
from warmup_scheduler import GradualWarmupScheduler


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.L1Loss()

    # ========== 优化器与调度器 ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    dataloader = train_dataloader(args.train_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)

    warmup_epochs = 3
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epoch - warmup_epochs, eta_min=1e-6
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
    )
    scheduler.step()

    # ========== 模型加载与断点恢复 ==========
    start_epoch = 1
    best_psnr = -1

    # 优先支持 args.model_dir 作为预训练权重路径
    if getattr(args, "model_dir", None) and os.path.isfile(args.model_dir):
        print(f"Loading pretrained checkpoint from {args.model_dir} ...")
        checkpoint = torch.load(args.model_dir, map_location=device)

        # 模型参数
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        # 优化器参数
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

        # epoch 信息
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {checkpoint['epoch']}")
        else:
            print("No epoch info in checkpoint, starting from 1.")

        print("Checkpoint loaded successfully!\n")

    # 若用户通过 args.resume 指定断点文件
    elif getattr(args, "resume", None):
        if os.path.isfile(args.resume):
            print(f"Resuming training from {args.resume}")
            state = torch.load(args.resume, map_location=device)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_epoch = state['epoch'] + 1
            print(f"Resumed from epoch {state['epoch']}\n")

    # ========== TensorBoard 初始化 ==========
    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')

    # ========== 开始训练 ==========
    for epoch_idx in range(start_epoch, args.num_epoch + 1):
        epoch_timer.tic()
        iter_timer.tic()

        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)

            # 多尺度标签
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')

            # 内容损失 (L1)
            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)
            loss_content = l1 + l2 + l3

            # 频域损失 (FFT)
            label_fft1 = torch.fft.fft2(label_img4, dim=(-2, -1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)
            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2, -1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)
            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)
            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1 + f2 + f3

            # 总损失
            loss = loss_content + 0.1 * loss_fft
            loss.backward()

            # ✅ 保留论文中的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())
            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

            # 打印与记录
            if (iter_idx + 1) % args.print_freq == 0:
                # ✅ 保留论文使用的 scheduler.get_lr()
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0],
                    iter_pixel_adder.average(), iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(),
                                  iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(),
                                  iter_idx + (epoch_idx - 1) * max_iter)

                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()

        # ========== 保存最新模型 ==========
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_idx
        }, overwrite_name)

        # 定期保存
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, f'model_{epoch_idx}.pkl')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_idx
            }, save_name)

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" %
              (epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))

        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()

        # 验证阶段
        if epoch_idx % args.valid_freq == 0:
            val_psnr = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val_psnr))
            writer.add_scalar('PSNR', val_psnr, epoch_idx)
            if val_psnr >= best_psnr:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch_idx
                }, os.path.join(args.model_save_dir, 'Best.pkl'))
                best_psnr = val_psnr

    # ========== 保存最终模型 ==========
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)
