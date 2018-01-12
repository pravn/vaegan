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
An additional term is added to the reconstruction loss (originally, mean squared error) which is obtained from a GAN.
I am not completely sure if this is the correct implementation (the paper has a third, additional term for z_p).
However, as implemented here, it boils down to some very simple code.

Samples - fake_samples.png and real_samples.png

