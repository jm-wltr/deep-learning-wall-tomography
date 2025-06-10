# Sonic Tomography with Deep Learning

This project explores the use of deep learning to reconstruct the internal structure of masonry walls using acoustic tomography data. Instead of relying on classical algebraic reconstruction techniques, we aim to train neural networks directly on wave propagation signals generated via simulation and measurement.

## 📁 Repository Structure
```
data/
├── waveforms/ # Raw Y-displacement waveforms from COMSOL (was "numerical analyses")
├── rays/ # Ray path metadata
├── sections/ # Ground-truth wall cross-section images for each simulation
docs/
└── data.md # Detailed documentation of the dataset structure and formats
```

## 📁 Dataset
The data we are working with was obtained via a COMSOL Multiphysics simulation. It is documented in [`docs/data.md`](docs/data.md).