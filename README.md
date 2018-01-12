#GAN
Adapts pytorch repo code to work with 28x28 data.

One needs to know how to work with transpose convolutions. 
Check this page. 
http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html#transposed-convolution-arithmetic

The notebook scratch.ipynb has some example attempts. 

#GAN VAE
An implementation attempt of the GAN+VAE paper 

Autoencoding beyond pixels using a learned similarity metric
https://arxiv.org/abs/1512.09300

This paper attempts to add a learned reconstruction loss to a VAE.
The VAE loss term for log p(x|z) in the decoder is replaced by a reconstruction loss
proportional to p(D(x)|D(x_tilde)) ~ N(D(x)|D(x_tilde)), where x is an input image, x_tilde is a
sample generated from the decoder Dec(z).
Therefore, I interpret D(x) = 1. 

We therefore seem to be minimizing (1-D(Dec(z)))^2.

This produces phantom digits that seem plausible, but there might be some generator/discriminator dynamics
that I am missing. 



Samples - fake_samples.png and real_samples.png

