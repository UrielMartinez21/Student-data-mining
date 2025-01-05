import numpy as np
import torch

def preprocess_input(escuela, materias_reprobadas, apoyo_familiar, apoyos_economicos, ejercer_carrera):
    escuela_mapping = {
        "ENCB":0,
        "ESCOM":1,
        "ESFM":2,
        "ESIA":3,
        "ESIME":4,
        "ESIQIE":5,
        "ESIT":6,
        "EST":7,
        "UPIEM":8
    }

    apoyos_economicos_mapping = {'Sí': 1, 'No': 0}
    ejercer_carrera_mapping = {'Sí': 2, 'No estoy seguro': 1, 'No': 0}

    # Encode input data
    escuela_encoded = escuela_mapping.get(escuela, -1)  # Return -1 if the key is not found
    apoyos_economicos_encoded = apoyos_economicos_mapping.get(apoyos_economicos, 0)
    ejercer_carrera_encoded = ejercer_carrera_mapping.get(ejercer_carrera, 0)

    # Create input tensor
    input_data = np.array([
        escuela_encoded,
        materias_reprobadas,
        apoyo_familiar,
        apoyos_economicos_encoded,
        ejercer_carrera_encoded
    ])

    return torch.tensor(input_data, dtype=torch.float32)


def calculate_metrics(y_true, y_pred):
    epsilon = 1e-8  # Pequeño valor para evitar divisiones por cero

    # Filtrar valores donde y_true es cero para MAPE
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.numel() == 0:  # Verificar si el tensor quedó vacío después del filtrado
        return float('nan'), float('nan'), float('nan'), float('nan')

    # Cálculo de métricas
    mse = torch.mean((y_true - y_pred) ** 2)  # MSE
    rmse = torch.sqrt(mse)  # RMSE
    mae = torch.mean(torch.abs(y_true - y_pred))  # MAE
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100  # MAPE con estabilidad numérica

    return mse.item(), rmse.item(), mae.item(), mape.item()


