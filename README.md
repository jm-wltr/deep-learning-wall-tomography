# Sonic Tomography with Deep Learning

This project explores the use of deep learning to reconstruct the internal structure of masonry walls using acoustic tomography data. Instead of relying on classical algebraic reconstruction techniques, we aim to train neural networks directly on wave propagation signals generated via simulation and measurement. The previous version of the project I am building on can be found [here](https://saco.csic.es/s/k5ty8eazD85pd4M).

## 📁 Repository Structure
```
├── common/ # Shared utilities and data loaders
│ ├── config.py # Paths, device, global constants
│ ├── dmatrix.py # Ray-voxel intersection logic
│ ├── pmatrix.py # Section image → purity mask
│ ├── waveforms.py # Waveform I/O & reduction
│ └── single_param_dataset.py # Not working yet!
├── models/ # Neural network definitions
│ └── autoencoder/ # Autoencoder dataset & architectures
│ └── pixel/ # Pixel-based classifiers
│ └── image/ # Full-image convolutional models
├── scripts/ # Entry-point scripts (training, evaluation)
│ └── train_and_eval.py # Not done yet!
├── experiments/ # Versioned experiment pipelines
│ ├── v1_baseline/ # Ray-metadata experiments
│ └── v2_autoencoder/ # Autoencoder experiments
├── notebooks/ # Exploratory analyses and prototyping
│ ├── matrices_demo.ipynb # Demo of dmatrix and pmatrix
│ └── waveforms.ipynb # Demo of waveforms.py
├── data/ # Raw and preprocessed data
│ ├── waveforms/ # Y-displacement waveforms from COMSOL
│ ├── rays/ # Ray path metadata (rayXX.txt)
│ └── sections/ # Wall cross-section images (XX.jpg)
├── artifacts/ # Generated intermediate files (e.g. AE .pt)
├── results/ # Model outputs, figures, TensorBoard logs
├── docs/ # Documentation
│ ├── data.md # Dataset formats and conventions
│ └── tests.md # Testing instructions
├── README.md # This file
└── requirements.txt # pip install dependencies
```

## 📁 Dataset
The data we are working with was obtained via a COMSOL Multiphysics simulation. It is documented in [`docs/data.md`](docs/data.md).

## 📁 Versions
To obtain the tomography images, we have been experimenting with two different methods. The first one is to obtain the images based on the sonic rays metadata; and the second one is to use the raw waveforms instead.

# Instructions

```
# 1. Create a venv folder named “.venv”
python -m venv .venv

# 2. Activate it
# • Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Pip install requirements (these are generated with `pip freeze > requirements.txt`)
pip install -r requirements.txt

```

To run tests, see [docs/tests.md](docs/tests.md).