import os
import torch
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
WAVES_DIR = os.path.join(DATA_DIR, 'waveforms')


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("config.py: DEVICE is set as " + str(DEVICE))

# Simulation domain parameters
dims = [[0,0.6],[0,0.4],[0,0]]
resolution = 0.02  # in dm

# Global file locations
RAYS_DIR = os.path.join(DATA_DIR, 'rays')
SECTIONS_DIR = os.path.join(DATA_DIR, 'sections')

# Color mappings
colors = {
    "Piedra": [172, 96, 73],
    "Mortero": [37, 173, 221]
}

# Emitter and receiver positions
emitter_X_positions = np.linspace(0.05, 0.55, 6)  # 6 points → 6 values
emitter_R_positions = np.linspace(0.05, 0.55, 11) # 11 points → 11 values
emitter_YE: float = 0.4
emitter_YR: float = 0.0