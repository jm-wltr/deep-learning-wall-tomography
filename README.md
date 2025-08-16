# Sonic Tomography with Deep Learning

This project explores the use of deep learning to reconstruct the internal structure of masonry walls using acoustic tomography data. Instead of relying on classical algebraic reconstruction techniques, we aim to train neural networks directly on wave propagation signals generated via simulation and measurement. The previous version of the project I am building on can be found [here](https://saco.csic.es/s/k5ty8eazD85pd4M). This repository contains the files related to both the simulations and the neural networks, as well as a pregenerated small sample dataset for trying out the neural network.

## Repository Structure
```
├── artifacts  # PyTorch logs, datasets, and models
├── common/ # Shared utilities and data loaders
├── COMSOL/ # Physical simulation files
├── data/
├── docs/ # Detailed documentation
├── models/ 
│ └── autoencoder/ # Autoencoder dataset & architectures
│ └── pixel_nn/ # Pixel-by-pixel neural network dataset & architectures
├── notebooks/ # Jupyter Notebook demos
├── results/ # Model outputs, figures
├── sections_generator/ # Code to generate random wall cross-sections (jpeg and stl)
├── .gitignore
├── README.md # This file
```

## Understanding the project
The project follows a four-step workflow, with each step documented in detail in its own file within the `docs/` folder:

1. **[Cross-Section Generation](docs/sections_generator.md)**  
   Random wall cross-sections are procedurally generated to simulate different internal structures.

2. **[Wave Simulation](docs/comsol.md)**  
   The generated cross-sections are used in COMSOL to simulate acoustic wave propagation and collect synthetic waveform data.

3. **[Waveform Compression](docs/autoencoder.md)**  
   An autoencoder neural network is trained to compress the high-dimensional waveform data into a compact latent representation.

4. **[Cross-Section Prediction](docs/pixel_nn.md)**  
   A second neural network is trained to predict the wall cross-section image from the compressed waveforms.

## Instructions

It is best to run this code with NVIDIA GPU. In this case you will need a Python version between 3.9-3.12 so that Torch is compatible with cuda (GPU). Otherwise, it will run on CPU. For optimal compatibility, I have been using Python 3.10.11.
```
# 1. Create a Python virtual environment called “.venv”
py -3.10 -m venv .venv

# 2. Activate it
# • Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Pip install requirements
pip install opencv-python numpy matplotlib jupyterlab ipykernel pandas scipy ezdxf
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard
```

Have fun!