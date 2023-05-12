# LiFi-GAN: A Lightweight HiFi-GAN Variant Incorporating (Inverse) Short-time Fourier Transform

## Description
This repo contains code implementing my graduation thesis for undergraduate degree.

In short, LiFi-GAN is a [HiFi-GAN V1](https://arxiv.org/abs/2010.05646) variant, with a lighter V1-C8C8I generator from [iSTFTNet](https://arxiv.org/abs/2203.02395) and a lighter MFD discriminator from [Basis-MelGAN](https://arxiv.org/abs/2106.13419) replacing HiFi-GAN's MPD. It also uses STFT loss from Basis-MelGAN and dropped out feature loss as Basis-MelGAN mentioning it can make the network failing to converge.
In our testing, this model can have ~50% faster inference speed and ~50% faster training speed, while maintaining a pretty good audio quality.

`Disclaimer: This repo is build for testing purpose. The code is not optimized for performance, nor have being well commented and tested. Modifications are rushed in hurry so a lot of temporary solutions involved. It should not being considered production ready.`

## Enviroment requirement
It should stays same as [HiFi-GAN](https://github.com/jik876/hifi-gan). You may need to upgrade torch buliding against newer cuda version for newer GPU though.

## Train
Please refer to [Official HiFi-GAN implementation](https://github.com/jik876/hifi-gan) for more info. In short, use this:
```
python3 train.py --config config_v1.json
```

## Inference
Please refer to [Official HiFi-GAN implementation](https://github.com/jik876/hifi-gan) for more info. In short, use this:
```
python3 inference.py --input_wavs_dir=[where to find input wav] --output_dir=[where to output] --checkpoint_file=[checkpoint file]
```
We also proivide a extension parameter `--device=[cuda or cpu]` here.

## Pretrained model
A copy of generator checkpoint can be retrieved here: [link](https://1drv.ms/u/s!Ar0Z7EPFmhMDn-BpY29LbhYD6p7kaQ?e=z371UM), which is trained on LJSpeech dataset with 1M steps. 

## References
This repo is based on:
- [Official HiFi-GAN implementation](https://github.com/jik876/hifi-gan)
- [rishikksh20's iSTFTNet implementation](https://github.com/rishikksh20/iSTFTNet-pytorch)
- [Official Basis-MelGAN implementation](https://github.com/xcmyz/FastVocoder)
