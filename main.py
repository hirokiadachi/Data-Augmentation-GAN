import os
import yaml
import shutil
import argparse
import numpy as np
import multiprocessing
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from models import Generator, Discriminator

p = argparse.ArgumentParser()
p.add_argument('--cfile', '-c', type=str, default='config')
p.add_argument('--checkpoints', type=str, default='checkpoints')
p.add_argument('--gpu', '-g', type=str, default='0',
               help='# of GPU. (1 GPU: single GPU)')
p.add_argument('--resume', '-r', type=str, default='')
p.add_argument('--seed_pytorch', type=int, default=np.random.randint(4294967295))
p.add_argument('--seed_numpy', type=int, default=np.random.randint(4294967295))
args = p.parse_args()
np.random.seed(args.seed_numpy)
torch.manual_seed(args.seed_pytorch)

##################################
# Loading training configure
##################################
with open(args.cfile) as yml_file:
    config = yaml.safe_load(yml_file.read())['training']

epochs = config['epochs']
batch_size = config['batch_size']
lr = config['lr']
beta = config['beta']
weight_decay = config['weight_decay']
tb = config['tb']
img_size = config['img_size']
num_classes = config['num_classes']
z_dim = config['z_dim']
dataset = config['dataset']
dropout_rate = config['dropout_rate']
gp_lambda = config['gp_lambda']

print('#'*50)
print('# Batch size: {}\n'
      '# Epoch: {}\n'
      '# Learning rate: {}\n'
      '# Beta1/Beta2: {}/{}\n'
      '# Weight decay: {}\n'
      '# Image size: {}\n'
      '# Number of classes: {}\n'
      '# Dropout rate: {}\n'
      '# Dataset: {}'.format(batch_size, epochs, lr, beta[0], beta[1], weight_decay, img_size, num_classes, dropout_rate, dataset))
print('#'*50)

os.makedirs(args.checkpoints, exist_ok=True)
tb_path = os.path.join(args.checkpoints, tb)
if os.path.exists(tb_path):    shutil.rmtree(tb_path)
tb = SummaryWriter(log_dir=tb_path)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda:0')

def main():
    iters = 0
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    train_data = datasets.__dict__[dataset.upper()]('/root/mnt/datasets/data', train=True, download=True, transform=train_transform)
    test_data = datasets.__dict__[dataset.upper()]('/root/mnt/datasets/data', train=False, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, num_workers=multiprocessing.cpu_count())
    
    G = nn.DataParallel(Generator(channels=3, dropout_rate=dropout_rate).to(device))
    D = nn.DataParallel(Discriminator(channels=3*2, dropout_rate=dropout_rate).to(device))
    
    G_opt = optim.Adam(G.parameters(), lr=lr, betas=(beta[0], beta[1]))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(beta[0], beta[1]))
    
    for epoch in range(epochs):
        iters = train(epoch, G, D, G_opt, D_opt, train_loader, iters)
        test(epoch, G, test_loader)
        checkpoint = {"numpy_seed": args.seed_numpy,
                      "torch_seed": args.seed_pytorch,
                      "gen": G.state_dict(),
                      "dis": D.state_dict(),
                      "gen_opt": G_opt.state_dict(),
                      "dis_opt": D_opt.state_dict()}
        torch.save(checkpoint, os.path.join(args.checkpoints, 'checkpoint.pth.tar'))

def gradient_penalty(D, real1, real2, fake):
    #real1 = Variable(real1, requires_grad=True)
    eps = torch.rand(real2.size(0),1,1,1).expand_as(real2).to(device)
    interpolated = Variable(eps * real2.data + (1 - eps) * fake.data, requires_grad=True)
    out = D(real1, interpolated)[0]
    grad = torch.autograd.grad(outputs=out,
                               inputs=interpolated,
                               grad_outputs=torch.ones(out.size()).to(device),
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0].view(out.size(0), -1)
    grad_l2norm = torch.sqrt(torch.sum(grad**2, dim=1))
    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)
    return d_loss_gp
        
def train(epoch, G, D, G_opt, D_opt, train_loader, iters):
    G.train()
    D.train()
    for idx, (img, tgt) in enumerate(train_loader):
        iters += 1
        img1 = img.to(device)
        rand_index = torch.randperm(len(img1))
        img2 = img[rand_index].to(device)
        
        z = torch.randn(img1.size(0), z_dim).to(device)
        fake = G(img1, z)
        dr_out = D(img1, img2)
        df_out = D(img1, fake)
        
        gp_loss = gradient_penalty(D, img1, img2, fake)
        
        D_opt.zero_grad()
        d_loss = df_out.mean() - dr_out.mean() + gp_lambda*gp_loss
        d_loss.backward()
        
        D_opt.step()
        
        #g_loss = None
        if idx % 5 == 0:
            G_opt.zero_grad()
            z = torch.randn(img1.size(0), z_dim).to(device)
            fake = G(img1, z)
            dg_out = D(img1, fake)
            g_loss = -dg_out.mean()
            g_loss.backward()
            G_opt.step()
            
            tb.add_scalars('loss', {"dis": d_loss.item(), "gen": g_loss.item()}, iters)
            tb.add_scalar('gradient penalty', gp_loss.item(), iters)
            
        if idx % 100 == 0:
            print('Training epoch: {} [{}/{} ({:.0f}%)] | D loss : {:.6f} | G loss: {:.6f} | GP: {:.6f}|'\
                    .format(epoch, idx * len(img), len(train_loader.dataset),
                    100. * idx / len(train_loader), d_loss.item(), g_loss.item(), gp_loss.item()))

    return iters

def test(epoch, G, test_loader):
    G.eval()
    test_iter = iter(test_loader)
    imgs, tgt = test_iter.next()
    imgs = imgs.to(device)
    z = torch.randn(imgs.size(0), z_dim).to(device)
    fake = G(imgs, z)
    
    tb.add_images('Real images', imgs, global_step=epoch)
    tb.add_images('Generated images', fake, global_step=epoch)
    
if __name__ == '__main__':
    main()
    
        

