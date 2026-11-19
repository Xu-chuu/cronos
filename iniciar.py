import os
import webbrowser
import subprocess

# 1. Obtener la ruta exacta de tu interfaz web
ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_frontend = os.path.join(ruta_actual, "frontend", "index.html")

# 2. Abrir la interfaz automáticamente en tu navegador predeterminado
print("🌐 Abriendo la interfaz visual en tu navegador...")
webbrowser.open(f"file:///{ruta_frontend}")

# 3. Encender el servidor backend (FastAPI) para que la IA responda
print("🚀 Encendiendo el servidor de Cronos AI...")
try:
    subprocess.run(["py", "-m", "uvicorn", "backend.api:app", "--reload"])
except KeyboardInterrupt:
    print("\n🛑 Servidor apagado correctamente.")