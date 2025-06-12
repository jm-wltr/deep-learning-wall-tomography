from common.datasets import DatasetConUnParametro
from models.modelos import ModeloDropout
from scripts.train_and_eval import run_full_pipeline

# Prepare dataset
dataset = DatasetConUnParametro("Velocidad", binarizada=True)

# Define model configurations
configs = [([66,32,8], 64, 0.001, 0.0, 1),
           ([128,64,16], 64, 0.001, 0.0, 2),
           # ... more configs
          ]

models = []
for capas, bs, lr, drop, idx in configs:
    m = ModeloDropout(capas, bs, lr, drop, idx, carpeta=str(idx)+"/", desc="desc...")
    models.append(m)

# Run all
for m in models:
    run_full_pipeline(m, dataset, n_epochs=200)
