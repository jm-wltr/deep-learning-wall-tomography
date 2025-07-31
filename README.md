# Sonic Tomography with Deep Learning

This project explores the use of deep learning to reconstruct the internal structure of masonry walls using acoustic tomography data. Instead of relying on classical algebraic reconstruction techniques, we aim to train neural networks directly on wave propagation signals generated via simulation and measurement. The previous version of the project I am building on can be found [here](https://saco.csic.es/s/k5ty8eazD85pd4M). This repository contains the files related to both the simulations and the neural networks, as well as a pregenerated small sample dataset for trying out the neural network.

## 📁 Repository Structure
```
├── artifacts  # PyTorch logs, datasets, and models
├── common/ # Shared utilities and data loaders
├── COMSOL/ # Physical simulation files
├── data/
├── docs/ # Detailed documentation
├── models/ 
│ └── autoencoder/ # Autoencoder dataset & architectures
│ └── pixel_nn/ # Pixel-by-pixel neural network dataset & architectures
├── results/ # Model outputs, figures
├── sections_generator/ # Code to generate random wall cross-sections (jpeg and stl)
├── .gitignore
├── README.md # This file
```

## Understanding the project
The workflow consists of [(1)](docs/sections_generator.md) generating random wall cross sections, [(2)](docs/comsol.md) generating simulated wave data through the cross sections via COMSOL, [(3)](docs/autoencoder.md) training an autoencoder to compress the waveforms to a small number of values, and [(4)](docs/pixel_nn.md) training the neural network that predicts the cross section image based off of the autoencoded waveforms.

Each of these sections is documented in detail in separate files in the `docs` folder.

## Instructions

It is best to run this code with NVIDIA GPU. In this case you will need a Python version between 3.9-3.12 so that Torch is compatible with cuda. Otherwise, it will run on CPU. For optimal compatibility, I have been using Python 3.10.11.
```
# 1. Create a venv folder named “.venv” (you can also use py -3.10 instead of python)
python -m venv .venv

# 2. Activate it
# • Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Pip install requirements
pip install opencv-python numpy matplotlib jupyterlab ipykernel pandas scipy ezdxf
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard

```

To run tests, see [docs/tests.md](docs/tests.md) (these are not exhaustive at all).

Have fun!