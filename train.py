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
from modules_tied import CVAE
from modules_tied import Aux
from modules_tied import NetD
from modules_tied import loss_function

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
    
    

def get_direct_gradient_penalty(netD, x, label, gamma, cuda):
    _,output = netD(x,label)
    gradOutput = torch.ones(output.size()).cuda() if cuda else torch.ones(output.size())
    
    gradient = torch.autograd.grad(outputs=output,
                                   inputs=x, grad_outputs=gradOutput,
                                   create_graph=True, retain_graph=True,
                                   only_inputs=True)[0]
    gradientPenalty = (gradient.norm(2, dim=1)).mean() * gamma
    
    return gradientPenalty


def run_trainer(netG,netD,optimizerG,optimizerD,train_loader,bsz):
    input = torch.FloatTensor(bsz,28,28)
    label = torch.FloatTensor(bsz)
    real_label=1
    fake_label=0
    USE_CUDA=1
    lamb = 1e-3
    l1dist = nn.PairwiseDistance(1)
    l2dist = nn.PairwiseDistance(2)
    LeakyReLU = nn.LeakyReLU(0)

    if(USE_CUDA):
        netG=netG.cuda()
        netD=netD.cuda()
        #aux = aux.cuda()
        #criterion=criterion.cuda()
        input,label=input.cuda(), label.cuda()
        l1dist = l1dist.cuda()
        l2dist = l2dist.cuda()
        LeakyReLU = LeakyReLU.cuda()

        e_fake = []
        e_enc  = []
        epochs = []

        label_vector_tmp = Variable(torch.zeros(bsz,10))

    for epoch in range(10000):
        e_fake_batch = 0
        e_enc_batch  = 0

        for i, (data,label) in enumerate(train_loader):
            gamma = 0.02
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

            for p in netD.parameters():
                p.requires_grad = True

            #need variables for dis
            #x_l, x_l_tilde
            #do discriminator calculations
            netD.zero_grad()


            #fc3_weight,fc4_weight = aux.return_weights()
            mu,logvar, fake = netG(inputv,label_vector)

            x_l_tilde, output_fake = netD(fake,label_vector)
            x_l, output_real = netD(inputv,label_vector)
            #x_l_aux, output_fake_aux = netD(fake_aux)

            pdist = l1dist(input.view(dataSize,-1), fake.view(dataSize,-1)).mul(lamb)

            #print('pdist.size()',pdist.size())
            errD_fake = LeakyReLU(output_real - output_fake + pdist).mean()
            #print('errD_fake.size()',errD_fake.size())

            errD_fake.backward(retain_graph=True)

            #gradient penalty
            #need to set gamma 
            #print('inputv.size()',inputv.size())
            gp = get_direct_gradient_penalty(netD,inputv,label_vector,10,True)
            gp.backward(retain_graph=True)

            optimizerD.step()


            for p in netD.parameters():
                p.requires_grad = False


            netG.zero_grad()

            L_enc = gamma*loss_function(x_l_tilde, x_l,mu,logvar)/bsz
            L_enc.backward()

            mu,logvar, fake = netG(inputv,label_vector)

            x_l_tilde, output_fake = netD(fake,label_vector)
            x_l, output_real = netD(inputv,label_vector)
            #x_l_aux, output_fake_aux = netD(fake_aux)

            pdist = l1dist(input.view(dataSize,-1), fake.view(dataSize,-1)).mul(lamb)

            #print('pdist.size()',pdist.size())
            errD_fake = -LeakyReLU(output_real - output_fake + pdist).mean()
            errD_fake.backward()

            gp = -get_direct_gradient_penalty(netD,inputv,label_vector,10,True)
            gp.backward(retain_graph=True)
            
            optimizerG.step()


            e_fake_batch += errD_fake.data.cpu().numpy()
            e_enc_batch += L_enc.data.cpu().numpy()


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
        epochs.append(epoch)
        e_fake.append(e_fake_batch)
        e_enc.append(e_enc_batch)
        plot_losses(epochs,e_fake,e_enc)

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

            
