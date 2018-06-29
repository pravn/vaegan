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

import matplotlib.pyplot as plt
import torch



#models
from CVAEWGAN import Encode
from CVAEWGAN import Decode
from CVAEWGAN import NetD

def plot_losses(epoch,e_fake_batch,e_enc_batch):
    plt.figure()
    plt.plot(epoch,e_fake_batch)
    plt.title('fake')
    plt.xlabel('epoch')
    #plt.show()
    plt.savefig('fake_loss.png')
    
    plt.close()

    plt.figure()
    print('e_enc', e_enc_batch)
    plt.plot(epoch,e_enc_batch)
    plt.title('encoder loss')
    plt.xlabel('epoch')
    plt.savefig('encoder_loss.png')
    plt.close()
    
def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def freeze_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False    

def run_trainer(netEnc,netDec,netD,optimizerEnc,
                optimizerDec,
                optimizerD,train_loader,bsz):
    input = torch.FloatTensor(bsz,28,28)
    label = torch.FloatTensor(bsz)
    real_label=1
    fake_label=0
    USE_CUDA=1
    lamb = 1e-3

    if(USE_CUDA):
        netEnc=netEnc.cuda()
        netDec=netDec.cuda()
        netDis=netD.cuda()
        #aux = aux.cuda()
        #criterion=criterion.cuda()
        input,label=input.cuda(), label.cuda()

        e_fake = []
        e_enc  = []
        epochs = []

        label_vector_tmp = Variable(torch.zeros(bsz,10))

    one = torch.Tensor(1)
    mone = torch.Tensor(-1)
    if torch.cuda.is_available():
        one = one.cuda()
        mone = mone.cuda()

    for epoch in range(10000):
        e_fake_batch = 0
        e_enc_batch  = 0

        for i, (data,label) in enumerate(train_loader):

            netEnc.zero_grad()
            netDec.zero_grad()
            netD.zero_grad()
            
            LAMBDA = 1.5
            real_cpu = data

            v = label.view(-1,1)
            label_vector_tmp.zero_()
            label_vector_tmp.scatter_(1,v,1)
            label_vector = Variable(label_vector_tmp).cuda()

            real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)

            dataSize = input.size(0)
            inputv = Variable(input,requires_grad=True)
            #irrelevant here 


            freeze_params(netEnc)
            freeze_params(netDec)
            free_params(netD)
            
            #need variables for dis
            #x_l, x_l_tilde
            #do discriminator calculations
            netD.zero_grad()


            mu_real = netEnc(inputv)
            mu_fake = torch.randn_like(mu_real)


            output_real = netD(mu_real,label_vector)
            output_fake = netD(mu_fake, label_vector)

            print('output_real.mean()', output_real.mean())
            print('output_fake.mean()', output_fake.mean())
            

            #maximize log(D(output_real)) + log(1-D(1-output_fake))
            #or minimize negative of that
            L_real = torch.log(output_real).mean()#minimize the negative 
            L_fake = torch.log(1-output_fake).mean() #...

            L_real.backward()
            L_fake.backward()
            
            optimizerD.step()


            free_params(netEnc)
            free_params(netDec)
            freeze_params(netD)
            
            z_real = netEnc(inputv)
            x_recon = netDec(z_real)

            d_real = netD(z_real)
            recon_loss = criterion(x_recon,inputv)

            recon_loss.backward(one)
            d_loss.backward(one)

            optimizerEnc.step()
            optimizerDec.step()


            #e_fake_batch += output_fake.data.cpu().numpy()
            #e_recon_batch += recon_loss.data.cpu().numpy()


            if (i % 100 == 0):
                print('epoch ', epoch)
            #print('real_cpu.size()', real_cpu.size())
                vutils.save_image(real_cpu,
                                './real_samples.png',
                                    normalize=True)
                vutils.save_image(fake.data.view(-1,1,28,28),
                                    './fake_samples.png',
                                    normalize=True)

            

        print('losses', epoch, e_fake_batch, e_enc_batch)
        #epochs.append(epoch)
        #e_fake.append(e_fake_batch)
        #e_enc.append(e_enc_batch)
        #plot_losses(epochs,e_fake,e_enc)

    def test(epoch):
        batch_size = 128
        model.eval()
        test_loss = 0

        label_vector_tmp = torch.FloatTensor(batch_size, 10)
    
        for i, (data, label) in enumerate(test_loader):
            if(data.size(0)!=128):
                break

        v = label.view(-1,1)
        label_vector_tmp.zero_()
        label_vector_tmp.scatter_(1,v,1)
        label_vector = Variable(label_vector_tmp)

        
        data = data.view(-1, data_size)
        
        if args.cuda:
            data = data.cuda()
            label_vector = label_vector.cuda()
            
        data = Variable(data, volatile=True)
        
        recon_batch, mu, logvar = model(data, label_vector)

        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data.view(batch_size, 1, 28, 28)[:n],
                                  recon_batch.view(batch_size, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

            
