import os
import json
import joblib
import pandas as pd
import logging
from preprocess import calcular_dias_mantenimiento

# Variables globales para los modelos
def init():
    global m_falla, m_subfalla, m_solucion, m_gravedad
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Azure monta los archivos en esta variable de entorno
    base_path = os.getenv("AZUREML_MODEL_DIR")
    
    # Lógica de carga robusta
    try:
        # Intento 1: Estructura estándar (carpeta model dentro del deploy)
        m_falla = joblib.load(os.path.join(base_path, "model", "chain_1_falla.pkl"))
        m_subfalla = joblib.load(os.path.join(base_path, "model", "chain_2_subfalla.pkl"))
        m_solucion = joblib.load(os.path.join(base_path, "model", "chain_3_solucion.pkl"))
        m_gravedad = joblib.load(os.path.join(base_path, "model", "chain_4_gravedad.pkl"))
        logger.info("✅ Cadena de 4 modelos cargada desde /model.")
        
    except Exception as e1:
        logger.warning(f"⚠️ No se encontró en /model, intentando raíz: {e1}")
        try:
            # Intento 2: Si Azure descomprimió los archivos en la raíz
            m_falla = joblib.load(os.path.join(base_path, "chain_1_falla.pkl"))
            m_subfalla = joblib.load(os.path.join(base_path, "chain_2_subfalla.pkl"))
            m_solucion = joblib.load(os.path.join(base_path, "chain_3_solucion.pkl"))
            m_gravedad = joblib.load(os.path.join(base_path, "chain_4_gravedad.pkl"))
            logger.info("✅ Cadena cargada desde raíz.")
        except Exception as e2:
            logger.error(f"🔥 ERROR FATAL: No se pudieron cargar los modelos. {e2}")
            raise e2

def run(raw_data):
    try:
        # 1. Convertir JSON a DataFrame
        data_dict = json.loads(raw_data)
        
        # Soporte para enviar un solo objeto o una lista "data": []
        if 'data' in data_dict:
            df = pd.DataFrame(data_dict['data'])
        else:
            df = pd.DataFrame([data_dict])
        
        # 2. Preprocesamiento (Cálculo de días)
        df = calcular_dias_mantenimiento(df)

        # 3. Features Base (SIN SENSORES)
        base_features = [
            'marca', 
            'modelo', 
            'anio', 
            'kilometraje', 
            'descripcion_sintomas', 
            'dias_ultimo_mant'
        ]
        
        # Asegurar tipos de datos de entrada
        df['kilometraje'] = df['kilometraje'].astype(int)
        df['anio'] = df['anio'].astype(int)
        # Asegurar que los síntomas sean string (evita error si llega nulo)
        df['descripcion_sintomas'] = df['descripcion_sintomas'].fillna("").astype(str)
        
        # -------------------------------------------------------
        # ⛓️ EJECUCIÓN EN CASCADA (Forzando Texto)
        # -------------------------------------------------------
        
        # PASO 1: Predecir Falla del Sistema
        pred_falla = m_falla.predict(df[base_features])[0][0]
        
        # PASO 2: Predecir Subfalla
        # IMPORTANTE: str() convierte la predicción a texto explícito
        df['falla_real'] = str(pred_falla) 
        features_step2 = base_features + ['falla_real']
        pred_subfalla = m_subfalla.predict(df[features_step2])[0][0]
        
        # PASO 3: Predecir Solución Técnica
        df['subfalla_real'] = str(pred_subfalla)
        features_step3 = features_step2 + ['subfalla_real']
        pred_solucion = m_solucion.predict(df[features_step3])[0][0]
        
        # PASO 4: Predecir Gravedad
        # AQUÍ OCURRÍA EL ERROR ANTES: Ahora forzamos a que sea texto
        df['solucion_real'] = str(pred_solucion)
        features_step4 = features_step3 + ['solucion_real']
        pred_gravedad = m_gravedad.predict(df[features_step4])[0][0]
        
        # Calcular confianza
        probs = m_falla.predict_proba(df[base_features])[0]
        confianza = max(probs)

        # 4. Respuesta
        return {
            "diagnostico_ia": {
                "sistema_afectado": pred_falla,
                "detalle_tecnico": pred_subfalla,
                "accion_recomendada": pred_solucion,
                "nivel_riesgo": pred_gravedad,
                "probabilidad_acierto": float(round(confianza, 2))
            },
            "meta": {
                "modelo": "v3_cascada_texto_forzado",
                "status": "success"
            }
        }

    except Exception as e:
        return {"error": f"Error en inferencia: {str(e)}"}