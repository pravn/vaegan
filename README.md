## GAN ##
Adapts pytorch repo code to work with 28x28 data.

One needs to know how to work with transpose convolutions. 
Check this page. 
http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic

The notebook scratch.ipynb has some example attempts. 

## GAN VAE ##
An implementation attempt of the GAN+VAE pape.

Autoencoding beyond pixels using a learned similarity metric
https://arxiv.org/abs/1512.09300

This paper attempts to add a learned reconstruction loss to a VAE.
The VAE loss term for log p(x|z) in the decoder is replaced by a reconstruction loss
proportional to $p(D(x)|D(x_\tilde)) \sim N(D_l(x)|D_l(\tilde{x}),I)$, where $x$ is an input image, $\tilde{x}$ is a
sample generated from the decoder Dec(z).

$D_l$ is the output of the $l$^{th} hidden layer of the discriminator. It is NOT the output of the discriminator itself. This is a crucial point in that the discriminator outputs a number between 0 and 1 after being passed into the sigmoid activation function. While it is conceivable that one could use this kind of setup, it utterly fails to reproduce the input. However, things improve after we pick the intermediate layer output (presumably, because it has features and is therefore more informative than 0, 1). The resulting loss function is MSE for these learned features. 

We therefore seem to be minimizing (D_l(x)-D_l(Dec(z)))^2.

Curiously, the paper has all the details laid out properly, although it is slightly obfuscated, or perhaps I was a little too dense to interpret it correctly.

The lines are as follows:

"To achieve this, let $Dis_l(x)$ denote the hidden representation of the $l$^{th} layer of the discriminator. We introduce a Gaussian observation model for $Dis_l(x)$ with mean $Dis_l(\tilde{x})$ and identity covariance

$p(Dis_l(x)|z) = \mathcal{N}(Dis_l(x)|Dis_l(\tilde{x}),I)$


Samples 
real_samples.png, fake_samples.png (conv arch)
real_fc_gen.png, fake_fc_gen.png (fully connected arch for generator decoder)


