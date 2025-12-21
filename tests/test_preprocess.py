import pandas as pd
import pytest
from datetime import datetime, timedelta
import sys
import os

# Agregamos la carpeta 'src' al sistema para poder importar tus funciones
# Esto es necesario porque la carpeta 'tests' está separada del código
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import calcular_dias_mantenimiento

def test_calculo_dias_normal():
    """Prueba que calcule bien los días cuando hay fecha válida"""
    hoy = datetime.now()
    hace_10_dias = hoy - timedelta(days=10)
    
    # Creamos un DataFrame falso de prueba
    df = pd.DataFrame({
        'ultimo_mantenimiento': [hace_10_dias]
    })
    
    # Ejecutamos tu función
    df_procesado = calcular_dias_mantenimiento(df)
    
    # Verificamos: El resultado debe ser aprox 10
    # (Usamos un rango pequeño por si hay diferencia de milisegundos)
    resultado = df_procesado['dias_ultimo_mant'].iloc[0]
    assert 9 <= resultado <= 11

def test_calculo_dias_nulo():
    """Prueba que devuelva -1 si el mecánico no puso fecha"""
    df = pd.DataFrame({
        'ultimo_mantenimiento': [None, pd.NaT]
    })
    
    df_procesado = calcular_dias_mantenimiento(df)
    
    # Verificamos: Debe ser -1 según tu regla de negocio
    assert df_procesado['dias_ultimo_mant'].iloc[0] == -1
    assert df_procesado['dias_ultimo_mant'].iloc[1] == -1