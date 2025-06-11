import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


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
Colores = {
    "Piedra": [172, 96, 73],
    "Mortero": [37, 173, 221]
}