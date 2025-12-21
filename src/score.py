import os
import json
import joblib
import pandas as pd
import logging
from preprocess import calcular_dias_mantenimiento

# Cargamos los 4 modelos en variables globales
def init():
    global m_falla, m_subfalla, m_solucion, m_gravedad
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    base_path = os.getenv("AZUREML_MODEL_DIR") # Carpeta en la nube
    # Si pruebas local, descomenta esto: base_path = "model" 
    
    # Cargamos la cadena completa
    try:
        m_falla = joblib.load(os.path.join(base_path, "model", "chain_1_falla.pkl"))
        m_subfalla = joblib.load(os.path.join(base_path, "model", "chain_2_subfalla.pkl"))
        m_solucion = joblib.load(os.path.join(base_path, "model", "chain_3_solucion.pkl"))
        m_gravedad = joblib.load(os.path.join(base_path, "model", "chain_4_gravedad.pkl"))
        logger.info("✅ Cadena de 4 modelos cargada correctamente.")
    except Exception as e:
        logger.error(f"🔥 Error cargando modelos: {e}")
        # Intento de ruta alternativa (a veces Azure duplica la carpeta model/model)
        try:
            path_fix = os.path.join(base_path, "chain_1_falla.pkl")
            if os.path.exists(path_fix):
                m_falla = joblib.load(os.path.join(base_path, "chain_1_falla.pkl"))
                m_subfalla = joblib.load(os.path.join(base_path, "chain_2_subfalla.pkl"))
                m_solucion = joblib.load(os.path.join(base_path, "chain_3_solucion.pkl"))
                m_gravedad = joblib.load(os.path.join(base_path, "chain_4_gravedad.pkl"))
                logger.info("✅ Cadena cargada (Ruta alternativa).")
            else:
                raise e
        except:
            raise e

def run(raw_data):
    try:
        data_dict = json.loads(raw_data)
        df = pd.DataFrame([data_dict])
        
        # 1. Preprocesamiento Base
        df = calcular_dias_mantenimiento(df)
        cols_sensores = [
            'sensor_rpm', 'sensor_presion_aceite', 'sensor_temperatura_motor', 
            'sensor_voltaje_bateria', 'sensor_velocidad', 'sensor_nivel_combustible'
        ]
        for col in cols_sensores:
            df[col] = df.get(col, pd.Series([-1] * len(df))).fillna(-1)

        # Definimos features base (orden correcto)
        base_features = [
            'marca', 'modelo', 'anio', 'kilometraje', 'descripcion_sintomas', 
            'dias_ultimo_mant'
        ] + cols_sensores
        
        # -------------------------------------------------------
        # ⛓️ EJECUCIÓN EN CASCADA
        # -------------------------------------------------------
        
        # PASO 1: Predecir Falla
        pred_falla = m_falla.predict(df[base_features])[0][0]
        
        # PASO 2: Predecir Subfalla
        # Inyectamos la predicción anterior como si fuera un dato real
        df['falla_real'] = pred_falla 
        features_step2 = base_features + ['falla_real']
        pred_subfalla = m_subfalla.predict(df[features_step2])[0][0]
        
        # PASO 3: Predecir Solución (Experto)
        # Inyectamos falla y subfalla
        df['subfalla_real'] = pred_subfalla
        features_step3 = features_step2 + ['subfalla_real']
        pred_solucion = m_solucion.predict(df[features_step3])[0][0]
        
        # PASO 4: Predecir Gravedad
        df['solucion_real'] = pred_solucion
        features_step4 = features_step3 + ['solucion_real']
        pred_gravedad = m_gravedad.predict(df[features_step4])[0][0]
        
        # Confianza (del primer modelo principal)
        confianza = max(m_falla.predict_proba(df[base_features])[0])

        # RETORNO FINAL
        return {
            "falla_predicha": pred_falla,
            "subfalla_predicha": pred_subfalla,
            "solucion_predicha": pred_solucion,
            "gravedad_predicha": pred_gravedad,
            "confianza": float(confianza),
            "mensaje": "Diagnóstico en Cascada Completado"
        }

    except Exception as e:
        return {"error": str(e)}