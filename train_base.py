import copy
import json
import os
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
import torch.distributed as dist

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet

device = torch.device('cuda:0')

FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'imagenet64'], help='dataset')
# flags.DEFINE_bool('train', False, help='train from scratch')
# flags.DEFINE_bool('eval', False, help='load model.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 1, 1], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1, 2], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 3, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.2, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_integer('T', 1024, help='total diffusion steps')
flags.DEFINE_integer('time_scale', 1, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'xstart', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
flags.DEFINE_enum('loss_type', 'both', ['x', 'eps', 'both'], help='loss type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('wd', 0.001, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_string('gpu_id', '4,5,6,7', help='multi gpu training')
flags.DEFINE_integer('local_rank', 0, help='local rank')
flags.DEFINE_bool('distributed', False, help='multi gpu training')
flags.DEFINE_integer('num_gpus', 4, help='multi gpu training')
flags.DEFINE_bool('conditional', False, help='use conditional or not')
flags.DEFINE_integer('class_num', 10, help='class num')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/CIFAR10/new_unet_eps/1024', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
# flags.DEFINE_bool('clip', True, help='clip when sampling')
# flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
# flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
# flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
# flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train():
    if get_rank() == 0:
        # if not os.path.exists(os.path.join(FLAGS.logdir, 'ddpm_clip')):
            # os.makedirs(os.path.join(FLAGS.logdir, 'ddpm_clip'))
        if not os.path.exists(os.path.join(FLAGS.logdir, 'ddim_clip')):
            os.makedirs(os.path.join(FLAGS.logdir, 'ddim_clip'))
        # if not os.path.exists(os.path.join(FLAGS.logdir, 'ddpm_noclip')):
            # os.makedirs(os.path.join(FLAGS.logdir, 'ddpm_noclip'))
        # if not os.path.exists(os.path.join(FLAGS.logdir, 'ddim_noclip')):
            # os.makedirs(os.path.join(FLAGS.logdir, 'ddim_noclip'))
    # dataset
    if FLAGS.dataset == 'cifar10':
        dataset = CIFAR10(
            root='../../data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif FLAGS.dataset == 'imagenet64':
        dataset = ImageFolder(
            '../../data/ImageNet64/train',
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    
    batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)
    if FLAGS.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=FLAGS.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                num_workers=FLAGS.num_workers, drop_last=True)

    # model setup
    net_model = UNet(
        T=FLAGS.T*FLAGS.time_scale, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout,
        conditional=FLAGS.conditional, class_num=FLAGS.class_num)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.T, FLAGS.time_scale, FLAGS.loss_type, FLAGS.mean_type).cuda(FLAGS.local_rank)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.T, FLAGS.time_scale, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).cuda(FLAGS.local_rank)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.T, FLAGS.time_scale, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).cuda(FLAGS.local_rank)
    if FLAGS.distributed:
        trainer = torch.nn.parallel.DistributedDataParallel(trainer, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank)
        net_sampler = torch.nn.parallel.DistributedDataParallel(net_sampler, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank)
        ema_sampler = torch.nn.parallel.DistributedDataParallel(ema_sampler, device_ids=[FLAGS.local_rank], output_device=FLAGS.local_rank)

    # log setup
    x_T = torch.randn(int(FLAGS.sample_size / FLAGS.num_gpus), 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.cuda(FLAGS.local_rank)
    grid = (make_grid(next(iter(train_loader))[0][:int(FLAGS.sample_size / FLAGS.num_gpus)]) + 1) / 2
    if get_rank() == 0:
        writer = SummaryWriter(FLAGS.logdir)
        writer.add_image('real_sample', grid)
        writer.flush()
        # backup all arguments
        with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
            f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    if get_rank() == 0:
        print('Model params: %.2f M' % (model_size / 1024 / 1024))

    # start training
    for step in range(1, FLAGS.total_steps + 1):
        if FLAGS.distributed:
            train_sampler.set_epoch(step)
        # train
        optim.zero_grad()
        samples = next(iter(train_loader))
        x_0, y = samples[0].cuda(FLAGS.local_rank), samples[1].cuda(FLAGS.local_rank)
        loss = trainer(x_0, y)
        torch.distributed.barrier()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            net_model.parameters(), FLAGS.grad_clip)
        optim.step()
        sched.step()
        ema(net_model, ema_model, FLAGS.ema_decay)

        # log
        if get_rank() == 0:
            writer.add_scalar('loss', loss, step)

        # sample
        if FLAGS.sample_step > 0 and (step % FLAGS.sample_step == 0 or step == 1):
            # net_model.eval()
            ema_model.eval()
            with torch.no_grad():
                y_target = torch.randint(FLAGS.class_num, size=(x_T.shape[0],), device=x_T.device)
                '''
                x_0 = ema_sampler(x_T, True, y=y_target)
                grid = (make_grid(x_0) + 1) / 2
                path = os.path.join(
                    FLAGS.logdir, 'ddpm_clip', '%d.png' % step)
                if get_rank() == 0:
                    save_image(grid, path)
                    writer.add_image('ddpm_clip', grid, step)
                '''
                x_0 = ema_sampler.module.ddim(x_T, 1, True, y=y_target)
                grid = (make_grid(x_0) + 1) / 2
                path = os.path.join(
                    FLAGS.logdir, 'ddim_clip', '%d.png' % step)
                if get_rank() == 0:
                    save_image(grid, path)
                    writer.add_image('ddim_clip', grid, step)
                '''
                x_0 = ema_sampler(x_T, False)
                grid = (make_grid(x_0) + 1) / 2
                path = os.path.join(
                    FLAGS.logdir, 'ddpm_noclip', '%d.png' % step)
                if get_rank() == 0:
                    save_image(grid, path)
                    writer.add_image('ddpm_noclip', grid, step)
                
                x_0 = ema_sampler.module.ddim(x_T, 1, False)
                grid = (make_grid(x_0) + 1) / 2
                path = os.path.join(
                    FLAGS.logdir, 'ddim_noclip', '%d.png' % step)
                if get_rank() == 0:
                    save_image(grid, path)
                    writer.add_image('ddim_noclip', grid, step)
                '''
            # net_model.train()
            ema_model.train()

        # save
        if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
            if get_rank() == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                    'T': trainer.module.T,
                    'time_scale': trainer.module.time_scale,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))
    if get_rank() == 0:
        writer.close()


def main(argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
    FLAGS.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    FLAGS.distributed = FLAGS.num_gpus > 1
    if FLAGS.distributed:
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    train()


if __name__ == '__main__':
    app.run(main)
