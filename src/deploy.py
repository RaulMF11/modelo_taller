import os
import sys
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration, OnlineRequestSettings
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# --- 1. CARGAR ENTORNO (FIX) ---
# Buscamos el .env en la misma carpeta donde está este script
current_dir = Path(__file__).resolve().parent
env_path = '.env'

print(f"🔍 Buscando .env en: {env_path}")
load_dotenv(dotenv_path=env_path)

# --- 2. CONFIGURACIÓN ---
# A. Credenciales (Desde el .env)
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")

# B. Configuración del Despliegue (FIJOS PARA ASEGURAR QUE NO SEAN NONE)
# Nota: Pon aquí el nombre exacto de tu endpoint que ya existe en Azure
ENDPOINT_NAME = "diag-exito-d9dd8" 

# El nombre que le pusimos al deployment (puede ser blue o green)
DEPLOYMENT_NAME = "blue" 

# El nombre EXACTO que salió en el paso anterior (train.py)
MODEL_NAME = "sistema-experto-completo" 

# El entorno que creamos la otra vez y que sabemos que funciona
ENV_NAME = "catboost-env-py310-final" 

def main():
    # Verificación de seguridad antes de arrancar
    if not SUBSCRIPTION_ID or not WORKSPACE_NAME:
        print("❌ ERROR: No se cargaron las credenciales del .env")
        return

    print(f"🚀 Iniciando despliegue en Endpoint: {ENDPOINT_NAME}")
    
    # 1. Conectar a Azure
    ml_client = MLClient(
        DefaultAzureCredential(), 
        SUBSCRIPTION_ID, 
        RESOURCE_GROUP, 
        WORKSPACE_NAME
    )
    
    # 2. Configurar el Despliegue
    # Azure buscará la ÚLTIMA versión (v2, v3...) automáticamente con @latest
    latest_model = f"{MODEL_NAME}@latest"
    latest_env = f"{ENV_NAME}@latest"
    
    print(f"📦 Usando Modelo: {latest_model}")
    print(f"🐍 Usando Entorno: {latest_env}")
    print(f"📂 Subiendo código desde: ./src")
    
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=latest_model,
        environment=latest_env,
        code_configuration=CodeConfiguration(
            code="./src", # Sube la carpeta src con score.py y preprocess.py
            scoring_script="score.py"
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
        request_settings=OnlineRequestSettings(request_timeout_ms=90000) 
    )
    
    # 3. Ejecutar
    print("⏳ Enviando instrucciones a Azure Cloud... (Esto tardará unos 8-10 minutos)")
    ml_client.begin_create_or_update(deployment).result()
    
    # 4. Asignar Tráfico (100% al nuevo despliegue)
    print("🚦 Actualizando tráfico al 100%...")
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.begin_create_or_update(endpoint).result()
    
    print(f"✅ ¡DESPLIEGUE COMPLETADO! Tu API está lista y actualizada.")

if __name__ == "__main__":
    main()