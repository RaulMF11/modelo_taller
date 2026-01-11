import os
import urllib
import pandas as pd
import joblib
from pathlib import Path
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from preprocess import calcular_dias_mantenimiento

# --- 1. CONFIGURACIÓN DE ENTORNO ---
current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / '.env'

print(f"🔍 Buscando .env en: {env_path}")
load_dotenv(dotenv_path=env_path)

# Variables de Azure y SQL
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID") 
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP") 
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME") 

SQL_SERVER = os.getenv("SQL_SERVER")
SQL_DB = os.getenv("SQL_DB")
SQL_USER = os.getenv("SQL_USER")
SQL_PWD = os.getenv("SQL_PWD")

def entrenar_modelo_cascada(df, X_cols, y_col, nombre_modelo, cat_features):
    """
    Función genérica para entrenar un eslabón de la cadena.
    """
    print(f"   ⚙️ Entrenando sub-modelo: {nombre_modelo} para predecir '{y_col}'...")
    
    # 1. Filtrar datos nulos en el objetivo
    df_train = df.dropna(subset=[y_col])
    
    # 2. Copiar X para evitar advertencias de pandas
    X = df_train[X_cols].copy()
    y = df_train[y_col]
    
    # --- CORRECCIÓN CRÍTICA: FORZAR TIPOS A STRING ---
    # Esto soluciona el error "Cannot convert to float" en Postman.
    # Aseguramos que CatBoost sepa que estas columnas SON TEXTO/CATEGORÍAS.
    if cat_features:
        for col in cat_features:
            if col in X.columns:
                # Rellenamos nulos con "Desconocido" y convertimos a string
                X[col] = X[col].fillna("Desconocido").astype(str)
    # -------------------------------------------------

    # Detección de columna de texto libre (NLP)
    text_features = []
    if 'descripcion_sintomas' in X_cols:
        text_features = ['descripcion_sintomas']
        # Aseguramos que el síntoma también sea string
        X['descripcion_sintomas'] = X['descripcion_sintomas'].fillna("").astype(str)

    # Configurar CatBoost
    model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        verbose=50,
        cat_features=cat_features,
        text_features=text_features,
        allow_writing_files=False
    )
    
    model.fit(X, y)
    
    path = f"model/{nombre_modelo}.pkl"
    joblib.dump(model, path)
    print(f"   ✅ Guardado en {path}")
    return model

def main():
    print("🚀 Iniciando Protocolo de Entrenamiento desde Azure SQL...")

    # --- 2. CONEXIÓN A BASE DE DATOS ---
    if not SQL_SERVER or not SQL_DB:
        raise ValueError("❌ Faltan variables de entorno SQL_SERVER o SQL_DB")

    print("🔌 Conectando a Azure SQL...")
    params = urllib.parse.quote_plus(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DB};"
        f"UID={SQL_USER};"
        f"PWD={SQL_PWD}"
    )
    conn_str = f"mssql+pyodbc:///?odbc_connect={params}"
    engine = create_engine(conn_str)
    
    # Traemos todo lo que tenga una 'falla_real' registrada
    query = "SELECT * FROM Diagnosticos WHERE falla_real IS NOT NULL"
    
    try:
        df = pd.read_sql(query, engine)
        print(f"📊 Registros cargados desde BD: {len(df)}")
    except Exception as e:
        print(f"❌ Error conectando a SQL: {e}")
        return

    if len(df) == 0:
        print("⚠️ ALERTA: La base de datos devolvió 0 registros con falla_real.")
        return

    # --- 3. PREPROCESAMIENTO ---
    print("🧹 Preprocesando datos...")
    
    # Calcular días desde último mantenimiento
    df = calcular_dias_mantenimiento(df)
    
    # Limpieza básica de nulos en columnas numéricas clave
    df['kilometraje'] = df['kilometraje'].fillna(0).astype(int)
    df['anio'] = df['anio'].fillna(2015).astype(int)

    # Creamos carpeta local para guardar modelos temporales
    os.makedirs("model", exist_ok=True)

    # ==============================================================================
    # ⛓️ ARQUITECTURA EN CASCADA (SIN SENSORES)
    # ==============================================================================
    
    # DEFINICIÓN DE FEATURES BASE
    base_features = [
        'marca', 
        'modelo', 
        'anio', 
        'kilometraje', 
        'descripcion_sintomas', 
        'dias_ultimo_mant'
    ]

    # --- FASE 1: FALLA (SISTEMA) ---
    print("\n🧠 [Nivel 1] Entrenando: Falla del Sistema...")
    entrenar_modelo_cascada(
        df, 
        X_cols=base_features, 
        y_col='falla_real', 
        nombre_modelo='chain_1_falla',
        cat_features=['marca', 'modelo']
    )

    # --- FASE 2: SUBFALLA (COMPONENTE) ---
    print("\n🧠 [Nivel 2] Entrenando: Componente Específico...")
    features_subfalla = base_features + ['falla_real'] 
    entrenar_modelo_cascada(
        df, 
        X_cols=features_subfalla, 
        y_col='subfalla_real', 
        nombre_modelo='chain_2_subfalla',
        cat_features=['marca', 'modelo', 'falla_real'] 
    )

    # --- FASE 3: SOLUCIÓN ---
    print("\n👨‍🔧 [Nivel 3] Entrenando: Generador de Soluciones...")
    features_solucion = base_features + ['falla_real', 'subfalla_real']
    entrenar_modelo_cascada(
        df,
        X_cols=features_solucion,
        y_col='solucion_real',
        nombre_modelo='chain_3_solucion',
        cat_features=['marca', 'modelo', 'falla_real', 'subfalla_real']
    )

    # --- FASE 4: GRAVEDAD ---
    print("\n⚠️ [Nivel 4] Entrenando: Calculadora de Riesgo...")
    # OJO: Aquí agregamos solucion_real como feature categórico, que era lo que fallaba
    features_gravedad = features_solucion + ['solucion_real']
    entrenar_modelo_cascada(
        df,
        X_cols=features_gravedad,
        y_col='gravedad_real',
        nombre_modelo='chain_4_gravedad',
        cat_features=['marca', 'modelo', 'falla_real', 'subfalla_real', 'solucion_real']
    )

    # --- 4. REGISTRO EN AZURE ---
    print("\n☁️ Subiendo modelos a Azure Machine Learning...")
    try:
        ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

        azure_model = Model(
            path="model", 
            name="sistema-experto-completo", 
            description="Modelo Cascada BD: Síntomas + Historial -> Diagnóstico",
            type=AssetTypes.CUSTOM_MODEL
        )

        registered_model = ml_client.models.create_or_update(azure_model)
        print(f"✅ ¡ÉXITO! Modelo registrado: {registered_model.name} (v{registered_model.version})")
        
    except Exception as e:
        print(f"⚠️ Error subiendo a Azure: {e}")

if __name__ == "__main__":
    main()