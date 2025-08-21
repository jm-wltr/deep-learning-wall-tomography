# Autoencoder for Waveforms

[Back to README.md](../README.md)

This document provides a concise explanation of the autoencoder implementation (located in `models/autoencoder`).

## 1. Purpose

A convolutional autoencoder is an unsupervised neural network that learns to compress (encode) input data into a low-dimensional latent representation and then reconstruct (decode) it back to the original domain. In this project:

* **Input**: 1D waveforms or sequences.
* **Encoder**: Series of Conv1d → (BatchNorm) → ELU → (Dropout) layers, followed by a bottleneck linear mapping to a latent vector.
* **Decoder**: Mirrors the encoder with linear → Unflatten → ConvTranspose1d layers to reconstruct the waveform.

With this, we can summarize our 10,000-point waveforms in 16 or 32 latent dimensions, which is helpful to reduce the data in training the tomography neural network.

## 2. Directory Structure

The file `dataset_autoencoder.py` implements a PyTorch dataset class that with which we wrap the data for the training process to be able to access it. The file `autoencoder_base.py` implements the base class for autoencoders (including training, logging, evaluating, saving and loading). The exact layer implementation is defined in `architectures/flexible_autoencoder`, and it has parameters that let you decide the number of latent dimensions, whether to use batchnorm and dropout... The script to run it is `scripts/train_flexible`, which can be run in the console as in the example (all parameters have reasonable defaults, see file):
```Usage:
    python -m models.autoencoder.scripts.train_flexible \
        --data-dir data/waveforms \
        --reduction resample --n 100 \
        --latent-dim 32 \
        --dropout 0.2 \
        --batchnorm \
        --batch-size 32 \
        --lr 1e-3 \
        --epochs 50 \
        --train-frac 0.8 \
        --seed 42 \
        --save-dir artifacts/autoencoder/checkpoints
```
This uses Adam and MSE. It will create log files in `artifacts/autoencoder` in the root folder. Most importantly, the checkpoints can be found there, with descriptive names. You can view the contents of the checkpoints (including live training visualization) via TensorBoard: 
```
tensorboard --logdir artifacts/autoencoder --port 6006
```
You will see the train and validation loss curves, and images of sample reconstructions of validation data every 10 epochs.

On the other hand, `dataset_autoencoder` defines the PyTorch dataset wrapping the waves, including reduction as specified in `waveforms.py` and illustrated in [notebooks\waveforms.ipynb](..\notebooks\waveforms.ipynb) (I recommend using resampling with n = 200). Finally, `utils` specifies small helper functions.

## 3. Experiment with results
To experiment with the models you train, you can use [notebooks/test_autoencoder.ipynb](../notebooks/test_autoencoder.ipynb), which will plot the loss curves, sample wave reconstructions from the validation dataset, the top 20 worst wave reconstructions, and a histogram of the loss per waveform. You should also take a look at it to get a good idea of how to load and use a trained autoencoder model. You can also see a compilation of plots from the notebook in [results/autoencoder/](../results/autoencoder/).