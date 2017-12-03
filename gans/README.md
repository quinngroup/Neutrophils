### Appearance Synthesis
---
This folder contains the code for synthesizing the appearance of neutrophils. It contains the following pieces:

1. implementation of DCGAN, WGAN, WGAN-GP for synthesizing gray-scale images of neutrophils. 
2. implementation of the evaluation technique: reconstruction of the test set
3. implementation of latent space walk

This code is adapted from: https://github.com/aosokin/biogans

### Usage
1. [code/](https://github.com/quinngroup/Neutrophils/tree/master/gans/code): Contains all the class files related to GANs. It contains the network architectures as well.
2. [experiments/](https://github.com/quinngroup/Neutrophils/tree/master/gans/experiments): Contains all the scripts to train the models.

To train the models from scratch, following scripts can be used. These scripts expect the path to the input dataset to be specified. 
```
sh experiments/train_64-64_gan-adam.sh

sh experiments/train_64-64_wgan-rmsprop.sh

sh experiments/train_64-64_wgangp-adam.sh
```

To run all the reconstruction experiments, following scripts can be used:
```
sh experiments/reconstruction_size-64-64_gan-adam.sh

sh experiments/reconstruction_size-64-64_wgan-rmsprop.sh

sh experiments/reconstruction_size-64-64_wgangp-adam.sh

```
After this, reconstruction errors can be viewed using [experiments/analyze_reconstruction_errors.ipynb](https://github.com/quinngroup/Neutrophils/blob/master/gans/experiments/analyze_reconstruction_errors.ipynb)

---
### Acknowledgments

1. Anton Osokin, Anatole Chessel, Rafael E. Carazo Salas and Federico Vaggi, GANs for Biological Image Synthesis, in proceedings of the International Conference on Computer Vision (ICCV), 2017.
