from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.utils as vutils


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


#def weights_init(m):
#    if classname.find('fc1


class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        #self.fc1 = nn.Linear(20, 784)
        #self.lrelu= nn.LeakyReLU(0.2, inplace=True)
        #self.tanh = nn.Tanh()

        self.main = nn.Sequential(
            #1x1->4x4
            nn.ConvTranspose2d(20,20*8,4,1,0,bias=False),  #(ic,oc,kernel,stride,padding)
            nn.BatchNorm2d(20*8), 
            nn.ReLU(True),
            nn.ConvTranspose2d(20*8,20*8,4,2,1,bias=False), #4x4->8x8
            nn.BatchNorm2d(20*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*8,20*8,4,2,1,bias=False), #8x8->16x16
            nn.BatchNorm2d(20*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*8,1,2,2,2,bias=False), #16x16->28x28
            nn.Tanh()
            )
        
            
            
            

    def forward(self,z):
        z = z.view(-1,z.size(1),1,1)
        return self.main(z)

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()
        self.main = nn.Sequential(
        #state size 1x28x28
            #28x28->16x16
            nn.Conv2d(1,8,2,2,2,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            #16x16->8x8
            nn.Conv2d(8,16,4,2,1,bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2,inplace=True),
            #8x8->4x4
            nn.Conv2d(16,32,4,2,1,bias=False),
            nn.BatchNorm2d(32),
            #4x4->1x1
            nn.Conv2d(32,1,4,1,0),
            nn.Sigmoid()
            )

    def forward(self,x):
        o = self.main(x.view(-1,1,28,28))
        return o
        

criterion = nn.BCELoss()

netG = _netG()
netD = _netD()

optimizerD = optim.Adam(netD.parameters(), lr = 1e-3)
optimizerG = optim.Adam(netG.parameters(), lr = 1e-3)

bsz = 100

input = torch.FloatTensor(bsz, 28, 28)
noise = torch.FloatTensor(bsz, 20).normal_(0,1)
fixed_noise = torch.FloatTensor(bsz, 20).normal_(0,1)

label = torch.FloatTensor(bsz)

real_label = 1
fake_label = 0

USE_CUDA = 1

if(USE_CUDA):
    netG = netG.cuda()
    netD = netD.cuda()
    #optimizerG = optimizerG.cuda()
    #optimizerD = optimizerD.cuda()
    criterion = criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise = noise.cuda()
    fixed_noise = fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

for epoch in range(10000):
    for i, (data,_) in enumerate(train_loader):
        #update D: maximize log((D(x)) + log(1-D(G(z)))
        netD.zero_grad()
        real_cpu = data
        
        if(USE_CUDA):
            real_cpu = real_cpu.cuda()

        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(bsz).fill_(real_label)

        inputv = Variable(input)
        labelv = Variable(label)

        print('real cpu', real_cpu.size())
        
        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        #train with fake
        noise.resize_(bsz, 20).normal_(0,1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        
        # update G: maximize log((D(G(z))
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, 100, i, len(train_loader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            print('real_cpu.size()', real_cpu.size())
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % args.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data.view(-1,1,28,28),
                    '%s/fake_samples.png' % (args.outf),
                    normalize=True)

    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))

    

