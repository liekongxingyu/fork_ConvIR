import os
import torch
import argparse
from torch.backends import cudnn
from models.ConvIR import build_net
from train import _train
from eval import _eval


def main(args):
    cudnn.benchmark = True

    # ===== 目录创建 =====
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # ===== 构建模型 =====
    model = build_net()
    print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    # ===== 模式选择 =====
    if args.mode == 'train':
        if args.model_dir and os.path.isfile(args.model_dir):
            print(
                f"🟢 Continue training from pretrained model: {args.model_dir}")
        elif args.resume and os.path.isfile(args.resume):
            print(f"🟢 Resume training from checkpoint: {args.resume}")
        else:
            print("🟠 Start training from scratch.")
        _train(model, args)

    elif args.mode == 'test':
        _eval(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ===== 基本参数 =====
    parser.add_argument('--model_name', default='ConvIR', type=str)
    parser.add_argument('--mode', default='train',
                        choices=['train', 'test'], type=str)
    parser.add_argument('--version', default='base',
                        choices=['small', 'base', 'large'], type=str)

    # ===== 数据集路径 =====
    parser.add_argument('--train_dir', type=str, default='../Dataset/Test')
    parser.add_argument('--val_dir', type=str,
                        default='../Dataset/RSCityscapes')

    # ===== 模型与训练控制 =====
    parser.add_argument('--model_dir', type=str, default='',
                        help='Path to pretrained model (optional)')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume training checkpoint (optional)')
    parser.add_argument('--test_model', type=str, default='',
                        help='Path to model for testing')
    parser.add_argument('--save_image', type=bool,
                        default=False, choices=[True, False])

    # ===== 训练参数 =====
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.5)

    args = parser.parse_args()

    # ===== 输出与结果目录 =====
    args.model_save_dir = os.path.join(
        'results', args.model_name, 'Training-Results')
    args.result_dir = os.path.join('results', args.model_name, 'test')
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # ===== 自动备份源码（论文中常见做法） =====
    os.system(f'cp models/layers.py {args.model_save_dir}')
    os.system(f'cp models/ConvIR.py {args.model_save_dir}')
    os.system(f'cp train.py {args.model_save_dir}')
    os.system(f'cp main.py {args.model_save_dir}')

    # ===== 打印配置 =====
    print("========== Configuration ==========")
    for k, v in vars(args).items():
        print(f"{k:20s}: {v}")
    print("==================================\n")

    main(args)
