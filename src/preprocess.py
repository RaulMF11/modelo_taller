import pandas as pd
import numpy as np
from datetime import datetime

def calcular_dias_mantenimiento(df):
    """
    Recibe un DataFrame con la columna 'ultimo_mantenimiento'.
    Devuelve el DF con una nueva columna 'dias_ultimo_mant'.
    Si es nulo, devuelve -1.
    """
    # Copiamos para no afectar el original
    df = df.copy()
    
    # Aseguramos que sea tipo datetime
    df['ultimo_mantenimiento'] = pd.to_datetime(df['ultimo_mantenimiento'], errors='coerce')
    
    fecha_actual = datetime.now()
    
    # Lógica vectorial (rápida): Restamos fecha actual - fecha mantenimiento
    df['dias_ultimo_mant'] = (fecha_actual - df['ultimo_mantenimiento']).dt.days
    
    # Llenamos los nulos con -1
    df['dias_ultimo_mant'] = df['dias_ultimo_mant'].fillna(-1)
    
    return df