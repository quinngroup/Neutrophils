#!/usr/bin/env bash

echo "Reconstruction GAN-ADAM"
sh ./reconstruction_size-64-64_gan-adam.sh
echo "Done GAN-ADAM"

echo "Reconstruction WGAN-RMSPROP"
sh ./reconstruction_size-64-64_wgan-rmsprop.sh
echo "Done WGAN-RMSPROP"

echo "Reconstruction WGANGP-ADAM"
sh ./reconstruction_size-64-64_wgangp-adam.sh
echo "Done WGANGP-ADAM"
