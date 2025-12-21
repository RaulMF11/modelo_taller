import os
import pandas as pd
import pyodbc
import joblib
from catboost import CatBoostClassifier
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from preprocess import calcular_dias_mantenimiento

# --- 1. CONFIGURACIÓN (¡LLENA ESTO!) ---
# Azure ML Details
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID") # Búscalo en el Portal de Azure (Overview)
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP") # El nombre de tu grupo de recursos
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME") # El nombre de tu recurso de Machine Learning

# Azure SQL Details
SQL_SERVER = os.getenv("SQL_SERVER")
SQL_DB = os.getenv("SQL_DB")
SQL_USER = os.getenv("SQL_USER")
SQL_PWD = os.getenv("SQL_PWD")

def entrenar_modelo_cascada(df, X_cols, y_col, nombre_modelo, cat_features):
    print(f"   ⚙️ Entrenando sub-modelo: {nombre_modelo} para predecir '{y_col}'...")
    
    X = df[X_cols]
    y = df[y_col]
    
    # Configuramos CatBoost
    model = CatBoostClassifier(
        # iterations=500,
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        verbose=50, # Silencio para no llenar la consola
        cat_features=cat_features,
        text_features=['descripcion_sintomas'] if 'descripcion_sintomas' in X_cols else None
    )
    
    model.fit(X, y)
    
    # Guardamos
    path = f"model/{nombre_modelo}.pkl"
    joblib.dump(model, path)
    print(f"   ✅ Guardado en {path}")
    return model

def main():
    print("🚀 Iniciando Protocolo de Entrenamiento en Cascada (CPMA)...")

    # --- 2. CARGA DE DATOS ---
    print("🔌 Conectando a Azure SQL...")
    conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SQL_SERVER};DATABASE={SQL_DB};UID={SQL_USER};PWD={SQL_PWD}'
    query = "SELECT * FROM Diagnosticos WHERE es_correcto = 1"
    df = pd.read_sql(query, pyodbc.connect(conn_str))
    
    # --- 3. PREPROCESAMIENTO ---
    print("🧹 Limpiando datos...")
    df = calcular_dias_mantenimiento(df)
    
    cols_sensores = [
        'sensor_rpm', 'sensor_presion_aceite', 'sensor_temperatura_motor', 
        'sensor_voltaje_bateria', 'sensor_velocidad', 'sensor_nivel_combustible'
    ]
    df[cols_sensores] = df[cols_sensores].fillna(-1)

    # Creamos carpeta para guardar los 4 cerebros
    os.makedirs("model", exist_ok=True)

    # ==============================================================================
    # ⛓️ ARQUITECTURA EN CASCADA (CHAINED MODELS)
    # ==============================================================================
    
    # Definimos las columnas base (Inputs del Mecánico)
    base_features = [
        'marca', 'modelo', 'anio', 'kilometraje', 'descripcion_sintomas', 
        'dias_ultimo_mant'
    ] + cols_sensores

    # ------------------------------------------------------------------
    # FASE 1: MODELO DIAGNÓSTICO (Identificar el problema)
    # ------------------------------------------------------------------
    
    # 1.1. Predecir FALLA PRINCIPAL
    # Input: Síntomas Base -> Output: Falla
    print("\n🧠 [Nivel 1] Entrenando Diagnosticador de Falla...")
    entrenar_modelo_cascada(
        df, 
        X_cols=base_features, 
        y_col='falla_real', 
        nombre_modelo='chain_1_falla',
        cat_features=['marca', 'modelo', 'descripcion_sintomas']
    )

    # 1.2. Predecir SUBFALLA
    # Input: Síntomas Base + FALLA (Usamos la real para enseñar/Teacher Forcing) -> Output: Subfalla
    print("\n🧠 [Nivel 1.5] Entrenando Diagnosticador de Subfalla...")
    # Truco: Agregamos la falla como input
    features_subfalla = base_features + ['falla_real'] 
    
    entrenar_modelo_cascada(
        df, 
        X_cols=features_subfalla, 
        y_col='subfalla_real', 
        nombre_modelo='chain_2_subfalla',
        cat_features=['marca', 'modelo', 'descripcion_sintomas', 'falla_real'] # falla_real es texto/categórico
    )

    # ------------------------------------------------------------------
    # FASE 2: MODELO EXPERTO CONTEXTUAL (Dar la Solución)
    # ------------------------------------------------------------------
    
    # 2.1. Predecir SOLUCIÓN
    # Input: Base + Falla + Subfalla -> Output: Solución
    # Aquí el modelo aprende: "Si es Freno (Falla) y Pastilla (Subfalla) y tiene 100k km, la solución es..."
    print("\n👨‍🔧 [Nivel 2] Entrenando Experto de Soluciones...")
    features_solucion = base_features + ['falla_real', 'subfalla_real']
    
    entrenar_modelo_cascada(
        df,
        X_cols=features_solucion,
        y_col='solucion_real',
        nombre_modelo='chain_3_solucion',
        cat_features=['marca', 'modelo', 'descripcion_sintomas', 'falla_real', 'subfalla_real']
    )

    # 2.2. Predecir GRAVEDAD
    # Input: Base + Falla + Subfalla + Solucion -> Output: Gravedad
    print("\n⚠️ [Nivel 2.5] Entrenando Analista de Gravedad...")
    features_gravedad = features_solucion + ['solucion_real']
    
    entrenar_modelo_cascada(
        df,
        X_cols=features_gravedad,
        y_col='gravedad_real',
        nombre_modelo='chain_4_gravedad',
        cat_features=['marca', 'modelo', 'descripcion_sintomas', 'falla_real', 'subfalla_real', 'solucion_real']
    )

    # --- 4. REGISTRO EN AZURE ---
    print("\n☁️ Subiendo el 'Cerebro Completo' a Azure...")
    ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

    azure_model = Model(
        path="model", # Subimos la carpeta con los 4 .pkl
        name="sistema-experto-completo", # Nuevo nombre para distinguir
        description="Sistema Cascada: Falla -> Subfalla -> Solucion -> Gravedad",
        type=AssetTypes.CUSTOM_MODEL
    )

    registered_model = ml_client.models.create_or_update(azure_model)
    print(f"✅ ¡ÉXITO! Modelo registrado: {registered_model.name} (v{registered_model.version})")

if __name__ == "__main__":
    main()