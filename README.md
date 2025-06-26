# Sonic Tomography with Deep Learning

This project explores the use of deep learning to reconstruct the internal structure of masonry walls using acoustic tomography data. Instead of relying on classical algebraic reconstruction techniques, we aim to train neural networks directly on wave propagation signals generated via simulation and measurement. The previous version of the project I am building on can be found [here](https://saco.csic.es/s/k5ty8eazD85pd4M).

## 📁 Repository Structure
```
├── common/ # Shared utilities and data loaders
├── models/ 
│ └── autoencoder/ # Autoencoder dataset & architectures
├── notebooks/ # Exploratory analyses and demos
├── data/ # Raw and preprocessed data
│ ├── waveforms/ # Y-displacement waveforms from COMSOL
│ ├── rays/ # Ray path metadata (rayXX.txt)
│ └── sections/ # Wall cross-section images (XX.jpg)
├── artifacts/ # Generated intermediate files (e.g. AE .pt)
├── results/ # Model outputs, figures
├── docs/ # Documentation
├── README.md # This file
```

## Understanding the codebase
The data we are working with was obtained via a COMSOL Multiphysics simulation. It is documented in [`docs/data.md`](docs/data.md). We pass the data through an Autoencoder and then another neural network to predict a tomography. For now I have implemented the Autoencoder, which is meant to compress waves with 10,000 points to a latent space of 16 or 32 points. Information about how it works, how to train it, and how to visualize its results is in [docs\autoencoder.md](docs\autoencoder.md). For the neural network itself, we already have the code to produce the 'ground truth' tomography images we want the model to predict based on waves, as well as for the DMatrix, which might help to correlate specific waves to specific pixels it might impact more in the neural network. This is all illustrated in [notebooks\matrices_demo.ipynb](notebooks\matrices_demo.ipynb). Finally, we have a few tests, just for dmatrix and pmatrix, and these are concisely explained in [docs/tests.md](docs/tests.md).

## Instructions

It is best to run this code with NVIDIA GPU. In this case you will need a Python version between 3.9-3.12 so that Torch is compatible with cuda. Otherwise, it will run on CPU. For optimal compatibility, I have been using Python 3.10.11.
```
# 1. Create a venv folder named “.venv” (you can also use py -3.10 instead of python)
python -m venv .venv

# 2. Activate it
# • Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Pip install requirements
pip install opencv-python numpy matplotlib jupyterlab ipykernel pandas scipy
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard

```

To run tests, see [docs/tests.md](docs/tests.md) (these are not exhaustive at all).

Have fun!