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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.mu_ = nn.Sequential(
            #28x28->12x12
            nn.Conv2d(1,8,5,2,0,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #12x12->4x4
            nn.Conv2d(8,64,5,2,0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #4x4->1x1: 20,1,1
            nn.Conv2d(64,20,4,1,0,bias=False),
            nn.ReLU(True)
            )


        self.logsigma_ = nn.Sequential(
            #28x28->12x12
            nn.Conv2d(1,8,5,2,0,bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            #12x12->4x4
            nn.Conv2d(8,64,5,2,0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #4x4->1x1: 20,1,1
            nn.Conv2d(64,20,4,1,0,bias=False),
            nn.ReLU(True)
            )
        

        self.dec_ = nn.Sequential(
            #1x1->4x4
            nn.ConvTranspose2d(20,20*8,4,1,0,bias=False),  #(ic,oc,kernel,stride,padding)
            nn.BatchNorm2d(20*8), 
            nn.ReLU(True),
            nn.ConvTranspose2d(20*8,20*16,4,2,1,bias=False), #4x4->8x8
            nn.BatchNorm2d(20*16),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*16,20*32,4,2,1,bias=False), #8x8->16x16
            nn.BatchNorm2d(20*32),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*32,1,2,2,2,bias=False), #16x16->28x28
            nn.Sigmoid()
            )
        

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def encode_new(self,x):
        return self.mu_(x), self.logsigma_(x)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu


    def decode_new(self,z):
        z = z.view(-1,z.size(1),1,1)
        return(self.dec_(z))
        
      
    def decode(self, z):
        z = z.view(-1,20)
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def dec_params(self):
        return self.fc3, self.fc4

    def forward(self, x):
        #mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode_new(x.view(-1, 1,28,28))
        z = self.reparameterize(mu, logvar)
        #return self.decode_new(z).view(-1,28,28), mu, logvar
        return self.decode(z).view(-1,28,28),mu,logvar



class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        #self.fc1 = nn.Linear(20, 784)
        #self.lrelu= nn.LeakyReLU(0.2, inplace=True)
        #self.tanh = nn.Tanh()

        self.main = nn.Sequential(
            #1x1->4x4
            nn.ConvTranspose2d(20,20*16,4,1,0,bias=False),  #(ic,oc,kernel,stride,padding)
            nn.BatchNorm2d(20*16), 
            nn.ReLU(True),
            nn.ConvTranspose2d(20*16,20*32,4,2,1,bias=False), #4x4->8x8
            nn.BatchNorm2d(20*32),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*32,20*64,4,2,1,bias=False), #8x8->16x16
            nn.BatchNorm2d(20*64),
            nn.ReLU(True),
            nn.ConvTranspose2d(20*64,1,2,2,2,bias=False), #16x16->28x28
            nn.Tanh()
            )
        
            
            
            

    def forward(self,z):
        z = z.view(-1,z.size(1),1,1)
        return self.main(z)

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy,self).__init__()

        self.fc3 = nn.Linear(20,400)
        self.fc4 = nn.Linear(400,784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def decode(self,z):
        z = z.view(-1,20)
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def dec_params(self):
        return self.fc3,self.fc4

    

    def forward(self,mu,logvar,other):
        self.fc3,self.fc4= other.dec_params()
        z = self.reparameterize(mu,logvar)
        other.fc3,other.fc4 = self.dec_params()
        return self.decode(z).view(-1,28,28)

        
        

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


        """
        self.main = nn.Sequential(
            nn.Conv2d(1,8,2,2,2,bias=False),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(8,16,4,2,1,bias=False),
            nn.LeakyReLU(0.2,inplace=True), #8x8
            nn.Conv2d(16,1,8,1,0), #1x1
            nn.Sigmoid()
            )"""

        """
        self.main = nn.Sequential(
            nn.Linear(784,1),
            nn.Sigmoid()
            )"""



    def forward(self,x):
        o = self.main(x.view(-1,1,28,28))
        #o = self.main(x.view(-1,784))
        return o


def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x.view(-1,784), x.view(-1, 784))
    #MSE = F.mse_loss(recon_x.view(-1,784), x.view(-1,784))
    MSE = F.mse_loss(recon_x,x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batch_size * 784

    return MSE + KLD


bsz = 100

def loss_new(gan_loss,mu,logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= bsz * 784

    return gan_loss + KLD
    


criterion = nn.BCELoss()

netG = VAE()
netD = _netD()
dummy = Dummy()

optimizerD = optim.Adam(netD.parameters(), lr = 1e-4)
optimizerG = optim.Adam(netG.parameters(), lr = 1e-4)
optimizer_dummy = optim.Adam(dummy.parameters(), lr = 1e-4)



input = torch.FloatTensor(bsz, 28, 28)
noise = torch.FloatTensor(bsz, 20).normal_(0,1)
fixed_noise = torch.FloatTensor(bsz, 20).normal_(0,1)

label = torch.FloatTensor(bsz)

real_label = 1
fake_label = 0

USE_CUDA = 1

if(USE_CUDA):
    #netG = torch.nn.DataParallel(netG).cuda()
    #netD = torch.nn.DataParallel(netD).cuda()
    netG = netG.cuda()
    netD = netD.cuda()
    dummy = dummy.cuda()
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
        dummy.zero_grad()
        
        real_cpu = data
        
        if(USE_CUDA):
            real_cpu = real_cpu.cuda()

        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(bsz).fill_(real_label)

        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        #train with fake
        fake,mu,logvar = netG(inputv)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()

        x_dummy = dummy(mu,logvar,netG)
        out_dummy = netD(x_dummy.detach())
        L_dummy = criterion(out_dummy, labelv)
        L_dummy.backward()
        
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake + L_dummy
        optimizerD.step()

        #step dummy
        x_dummy = dummy(mu,logvar,netG)
        out_dummy = netD(x_dummy.detach())
        labelv = Variable(label.fill_(real_label))
        LGAN_dummy = criterion(out_dummy,labelv)
        LGAN_dummy.backward()
        out_orig = netD(inputv)
        L_recon_dummy = loss_function(out_dummy.detach(),out_orig,mu.detach(),logvar.detach())
        L_recon_dummy.backward()
        optimizer_dummy.step()
        
        
        # update G: maximize log((D(G(z))
        if epoch%1==0:
            netG.zero_grad()
            labelv = Variable(label.fill_(real_label))
            output = netD(fake)
            output_orig = netD(inputv)
            #errG = criterion(output,labelv) + loss_function(fake,inputv,mu,logvar)
            errG = loss_function(output,output_orig,mu,logvar) #+ 0.5*criterion(output,labelv)


            errG.backward()
    
            D_G_z2 = output.data.mean()
            optimizerG.step()

        if(epoch %1 ==0):
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, 100, i, len(train_loader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                print('real_cpu.size()', real_cpu.size())
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % args.outf,
                                  normalize=True)
                vutils.save_image(fake.data.view(-1,1,28,28),
                                  '%s/fake_samples.png' % (args.outf),
                                  normalize=True)

    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))

    

