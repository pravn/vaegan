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
proportional to p(D(x)|D(x_tilde)) ~ N(D(x)|D(x_tilde),I), where x is an input image, x_tilde is a
sample generated from the decoder Dec(z).
D(x) is the output of data passed into discriminator
D(x_tilde) is the output of model (fake) passed into discriminator


We therefore seem to be minimizing (D(x)-D(Dec(z)))^2.

This produces phantom digits that seem plausible, but there might be some generator/discriminator dynamics
that I am missing. 


In the fully connected version of the generator (decoder), I let it run for a day to get samples that reconstruct zeros and ones,
but it has trouble with other digits, although the generated samples are most definitely reconstructions of digits, even if incorrectly reconstructed.
Moreover, styles aren't reconstructed properly either. Aspects such as slant and thickness are examples of stylistic considerations.

It is possible that I might have to include the additional term in the paper consisting of the ancillary generator with parameters tied with the encoder/primary generator.


Samples 
real_samples.png, fake_samples.png (conv arch)
real_fc_gen.png, fake_fc_gen.png (fully connected arch for generator decoder)


