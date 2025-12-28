import os
import urllib # <--- NUEVO
import pandas as pd
import joblib
from pathlib import Path
from sqlalchemy import create_engine # <--- NUEVO
from catboost import CatBoostClassifier
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from preprocess import calcular_dias_mantenimiento

# --- 1. CONFIGURACIÓN ROBUSTA DE ENTORNO ---
current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / '.env'

print(f"🔍 Buscando .env en: {env_path}")
load_dotenv(dotenv_path=env_path)

# --- DEBUG: VERIFICAR VARIABLES ---
server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_DB")

if not server or not database:
    print("❌ ERROR CRÍTICO: No se leyeron las variables del .env")
    exit()
else:
    print(f"✅ Variables cargadas. Servidor: {server} | BD: {database}")
    
# Azure ML Details
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID") 
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP") 
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME") 

# Azure SQL Details
SQL_SERVER = os.getenv("SQL_SERVER")
SQL_DB = os.getenv("SQL_DB")
SQL_USER = os.getenv("SQL_USER")
SQL_PWD = os.getenv("SQL_PWD")

def entrenar_modelo_cascada(df, X_cols, y_col, nombre_modelo, cat_features):
    print(f"   ⚙️ Entrenando sub-modelo: {nombre_modelo} para predecir '{y_col}'...")
    
    X = df[X_cols]
    y = df[y_col]
    
    model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        verbose=50,
        cat_features=cat_features,
        text_features=['descripcion_sintomas'] if 'descripcion_sintomas' in X_cols else None
    )
    
    model.fit(X, y)
    
    path = f"model/{nombre_modelo}.pkl"
    joblib.dump(model, path)
    print(f"   ✅ Guardado en {path}")
    return model

def main():
    print("🚀 Iniciando Protocolo de Entrenamiento en Cascada (CPMA)...")

    # --- 2. CARGA DE DATOS (FIX SQLAlchemy) ---
    print("🔌 Conectando a Azure SQL...")
    
    # Construimos la conexión segura que entiende fechas complejas
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DB};"
        f"UID={SQL_USER};"
        f"PWD={SQL_PWD}"
    )
    conn_str = f"mssql+pyodbc:///?odbc_connect={params}"
    engine = create_engine(conn_str) # Creamos el motor
    
    query = "SELECT * FROM Diagnosticos WHERE es_correcto = 1"
    
    # Leemos usando el engine (SQLAlchemy) en lugar de la conexión cruda
    df = pd.read_sql(query, engine) 
    
    # --- 3. PREPROCESAMIENTO ---
    print(f"🧹 Limpiando datos ({len(df)} registros encontrados)...")
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
    
    base_features = [
        'marca', 'modelo', 'anio', 'kilometraje', 'descripcion_sintomas', 
        'dias_ultimo_mant'
    ] + cols_sensores

    # FASE 1: FALLA
    print("\n🧠 [Nivel 1] Entrenando Diagnosticador de Falla...")
    entrenar_modelo_cascada(
        df, 
        X_cols=base_features, 
        y_col='falla_real', 
        nombre_modelo='chain_1_falla',
        cat_features=['marca', 'modelo', 'descripcion_sintomas']
    )

    # FASE 1.5: SUBFALLA
    print("\n🧠 [Nivel 1.5] Entrenando Diagnosticador de Subfalla...")
    features_subfalla = base_features + ['falla_real'] 
    entrenar_modelo_cascada(
        df, 
        X_cols=features_subfalla, 
        y_col='subfalla_real', 
        nombre_modelo='chain_2_subfalla',
        cat_features=['marca', 'modelo', 'descripcion_sintomas', 'falla_real'] 
    )

    # FASE 2: SOLUCIÓN
    print("\n👨‍🔧 [Nivel 2] Entrenando Experto de Soluciones...")
    features_solucion = base_features + ['falla_real', 'subfalla_real']
    entrenar_modelo_cascada(
        df,
        X_cols=features_solucion,
        y_col='solucion_real',
        nombre_modelo='chain_3_solucion',
        cat_features=['marca', 'modelo', 'descripcion_sintomas', 'falla_real', 'subfalla_real']
    )

    # FASE 2.5: GRAVEDAD
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
        path="model", 
        name="sistema-experto-completo", 
        description="Sistema Cascada: Falla -> Subfalla -> Solucion -> Gravedad",
        type=AssetTypes.CUSTOM_MODEL
    )

    registered_model = ml_client.models.create_or_update(azure_model)
    print(f"✅ ¡ÉXITO! Modelo registrado: {registered_model.name} (v{registered_model.version})")

if __name__ == "__main__":
    main()