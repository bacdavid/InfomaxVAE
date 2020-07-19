##### Disclaimer
This project is by David Bachmann (bacdavid). It is not published or affilated.

# InfomaxVAE

Obtain the latent variables that contain the maximal information wrt. sample (mutual information). This work is inspired by the InfoGAN (Chen et al., https://arxiv.org/abs/1606.03657) where the mutual information between selected channels and the sample is maximized. 

<div align="center">
<p>Single </p>
<img src="img/variable_perturbation.jpg" width="200"/>
</div>

## Details

### Variational Auto Encoder

- Typical VAE network for the generator: Encoder - Sampler - Decoder
- Mainly convolutional layers for the encoder and de-convolutional layers for the decoder with kernel size 5x5 and strides of 2x2
- Batch Norm followed by ReLU after the (de-)convolution
- 64 - 128 - 256, 256 - 128 - 64 - 3 (RGB) feature maps for encoder and decoder, respectively

### Latent Variables

- VAE-style latent variables are denoted by `z`
- InfomaxVAE-style latent variables by `c`

### Loss

The original VAE-loss is 
```
mse(x, x_vae) + E[ KL(p(z | x) || p(z)).
```
By adding the mutual information term, the following is obtained:<br/> 
*Please note that `[...;...]` denotes the concatenation operator*
```
mse(x, x_vae) + KL(p([z;c] | x) || p([z; c])) - I(x; c) 
= mse(x, x_vae) + KL(p([z;c] | x) || p([z; c])) - KL(p(c | x) || p(c))
= mse(x, x_vae) + KL(p(z | x) || p(z)).
```
In other words, exclude the InfomaxVAE-style latent variables from the regularization term.


## Try it

Simply open the file 
```
train.py
```
and perform the required adjustments.
