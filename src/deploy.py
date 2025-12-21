import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, CodeConfiguration, OnlineRequestSettings
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# --- CONFIGURACIÓN ---
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID") # <--- ¡PON EL TUYO!
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP") # El nombre de tu grupo de recursos
WORKSPACE_NAME = os.getenv("AZURE_WORKSPACE_NAME") # El nombre de tu recurso de Machine Learning

# ENDPOINT_NAME = "diag-exito-d9dd8" # Usamos el endpoint que YA existe para ahorrar
# DEPLOYMENT_NAME = "blue" # Reemplazaremos el despliegue actual
# MODEL_NAME = "sistema-experto-completo" # El nombre que registramos hoy
# ENV_NAME = "catboost-env-py310-final" # El entorno que ya funcionaba
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME") # Usamos el endpoint que YA existe para ahorrar
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME") # Reemplazaremos el despliegue actual
MODEL_NAME = os.getenv("MODEL_NAME") # El nombre que registramos hoy
ENV_NAME = os.getenv("ENV_NAME") # El entorno que ya funcionaba

def main():
    print(f"🚀 Iniciando despliegue en Endpoint: {ENDPOINT_NAME}")
    
    # 1. Conectar a Azure
    ml_client = MLClient(
        DefaultAzureCredential(), 
        SUBSCRIPTION_ID, 
        RESOURCE_GROUP, 
        WORKSPACE_NAME
    )
    
    # 2. Configurar el Despliegue
    # Azure buscará la ULTIMA versión de tu modelo automáticamente
    latest_model = f"{MODEL_NAME}@latest"
    latest_env = f"{ENV_NAME}@latest"
    
    print(f"📦 Usando Modelo: {latest_model}")
    print(f"🐍 Usando Entorno: {latest_env}")
    
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=latest_model,
        environment=latest_env,
        code_configuration=CodeConfiguration(
            code="./src", # Sube TODA la carpeta src (incluye preprocess.py y score.py)
            scoring_script="score.py"
        ),
        instance_type="Standard_DS3_v2",
        instance_count=1,
        request_settings=OnlineRequestSettings(request_timeout_ms=90000) 
    )
    
    # 3. Ejecutar (Esto tarda unos 8-10 minutos)
    print("⏳ Enviando instrucciones a Azure Cloud... (Esto puede tardar, ve por un café ☕)")
    ml_client.begin_create_or_update(deployment).result()
    
    # 4. Asignar Tráfico
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.begin_create_or_update(endpoint).result()
    
    print(f"✅ ¡DESPLIEGUE COMPLETADO! Tu API está lista.")

if __name__ == "__main__":
    main()