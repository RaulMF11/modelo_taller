import os
import sys
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration, OnlineRequestSettings
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# --- 1. CARGAR ENTORNO ---
# Ajuste: Buscamos el .env en la carpeta RAÍZ (padre de src), igual que en train.py
current_dir = Path(__file__).resolve().parent
env_path = current_dir.parent / '.env'

print(f"🔍 Buscando .env en: {env_path}")
load_dotenv(dotenv_path=env_path)

# --- 2. CONFIGURACIÓN ---
# A. Credenciales
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME")

# B. Configuración del Despliegue
ENDPOINT_NAME = "diag-exito-d9dd8"  # Tu endpoint existente
DEPLOYMENT_NAME = "blue"            # Estrategia Blue/Green
MODEL_NAME = "sistema-experto-completo" # ¡IMPORTANTE! Debe coincidir con train.py
ENV_NAME = "catboost-env-py310-final"   # Tu entorno funcional

def main():
    # Verificación de seguridad
    if not SUBSCRIPTION_ID or not WORKSPACE_NAME:
        print("❌ ERROR: No se cargaron las credenciales. Verifica la ruta del .env")
        return

    print(f"🚀 Iniciando actualización del Endpoint: {ENDPOINT_NAME}")
    
    # 1. Conectar a Azure
    ml_client = MLClient(
        DefaultAzureCredential(), 
        SUBSCRIPTION_ID, 
        RESOURCE_GROUP, 
        WORKSPACE_NAME
    )
    
    # 2. Configurar el Despliegue
    # Usamos @latest para que Azure tome automáticamente el modelo que acabas de entrenar
    latest_model = f"{MODEL_NAME}@latest"
    latest_env = f"{ENV_NAME}@latest"
    
    print(f"📦 Modelo seleccionado: {latest_model}")
    print(f"📂 Subiendo código (src/score.py + src/preprocess.py)...")
    
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=latest_model,
        environment=latest_env,
        code_configuration=CodeConfiguration(
            code="./src", # Importante: Sube toda la carpeta src para que score.py encuentre a preprocess.py
            scoring_script="score.py"
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
        # Aumentamos el timeout por si CatBoost tarda en cargar en memoria
        request_settings=OnlineRequestSettings(request_timeout_ms=90000) 
    )
    
    # 3. Ejecutar (Create or Update)
    print("⏳ Enviando despliegue a Azure Cloud... (Esto tardará unos 8-10 minutos)")
    ml_client.begin_create_or_update(deployment).result()
    
    # 4. Asignar Tráfico
    print("🚦 Redirigiendo el 100% del tráfico a la nueva versión...")
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.begin_create_or_update(endpoint).result()
    
    print(f"✅ ¡DESPLIEGUE COMPLETADO! Tu API está actualizada con la lógica de Tesis (Sin Sensores).")

if __name__ == "__main__":
    main()