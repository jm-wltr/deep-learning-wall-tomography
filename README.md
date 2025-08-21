# Sonic Tomography with Deep Learning

This project explores the use of deep learning as an alternative to classical algebraic techniques to reconstruct the internal structure of masonry walls using acoustic tomography data. The previous version of the project I am building on can be found [here](https://saco.csic.es/s/k5ty8eazD85pd4M). 

**What this does (at a glance):** We represent each wall cross-section as a 30×20 pixel grid with binary values (stone/mortar). For every section, an array of 6 emitters × 11 receivers produces 66 sonic “rays,” each yielding a 1D time-amplitude waveform. A convolutional autoencoder compresses each waveform to a small latent vector (e.g., 16 dims). For each pixel we then build a feature vector that mixes (a) the pixel’s real-world bounds, (b) per-ray geometry (emitter/receiver coordinates and path length through that pixel), and (c) the 66 latent waveforms. A lightweight MLP (“pixel_nn”) predicts, per pixel, the probability of **stone vs mortar** (or a continuous value), reconstructing the internal cross-section. We train on thousands of synthetic sections from COMSOL and evaluate on held-out data, effectively learning a data-driven inverse operator that can be more robust and easier to tune than classical algebraic solvers.

This repository contains the files needed for running both the simulations and the neural networks, as well as an example dataset with 300 wall cross-sections and their simulated blah blah for you to try out the neural network. We are currently working on generating much more data to

This repository contains everything needed to run the simulations and the neural networks, plus an example dataset with 300 wall cross-sections and their simulated waveforms so you can train/evaluate out of the box. We’re currently generating a much larger dataset that will be released separately. Once we have this dataset we will be able to properly evaluate our model and work on improving its architecture.

## Repository Structure
```
├── artifacts  # PyTorch logs, datasets, and models
├── common/ # Shared utilities and data loaders
├── COMSOL/ # Physical simulation files
├── data/ # Main dataset (for train/validate/test split)
├── data_held_out/ # Separate testing dataset
├── docs/ # Detailed documentation
├── models/ 
│ └── autoencoder/ # Autoencoder architectures
│ └── pixel_nn/ # Pixel-by-pixel neural network architectures
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
# (Optional). You can get Python 3.10.11 in Windows with this command
winget install --id Python.Python.3.10 --version 3.10.11

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