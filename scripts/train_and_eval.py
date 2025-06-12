from common.config import DEVICE

def run_full_pipeline(model, dataset, n_epochs, logs='Epochs'):
    print(f"\nRunning model {model.indice}\n")
    train_set, val_set, reserved = dataset.generarParticion()
    model.set_dataset(train_set, val_set, reserved)
    model.entrenar(n_epochs, shows=False, saves=True, logs=logs)
    model.save_all()
    model.prediccionesAleatorias(dataset, cmap="gray_r", filas=4, shows=False, saves=True)
