from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Esto permite que la API encuentre tus carpetas originales (pnl, entorno, agente)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnl.comprension import ComprensionLenguaje
from pnl.generador_groq import GeneradorRespuestasGroq
from entorno.entorno_dialogo import EntornoDialogo
from agente.q_learning import AgenteQLearning

app = FastAPI(title="API de Cronos")

# Permite que el HTML del frontend se comunique con este servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar los módulos de IA
pnl = ComprensionLenguaje("datos/intenciones.json", "datos/conocimiento.json")
entorno = EntornoDialogo(max_turnos=50)
agente = AgenteQLearning(num_acciones=len(entorno.ACCIONES))
agente.cargar_modelo("modelos/tabla_q.json")
generador = GeneradorRespuestasGroq()
estado_dialogo = entorno.reiniciar(modo_entrenamiento=False)

class Mensaje(BaseModel):
    texto: str

@app.post("/chatear")
def procesar_chat(mensaje: Mensaje):
    global estado_dialogo
    
    # 1. Comprensión (NLU)
    analisis = pnl.analizar_mensaje(mensaje.texto)
    estado_dialogo = entorno.procesar_mensaje_humano(analisis)
    
    # 2. Decisión del Agente (RL)
    indice_accion = agente.elegir_accion(estado_dialogo, en_entrenamiento=False)
    nombre_accion = entorno.ACCIONES[indice_accion]
    estado_dialogo, recompensa, terminado, info = entorno.paso(indice_accion)
    
    # 3. Generación Natural (NLG con Groq)
    respuesta = generador.generar(mensaje.texto, nombre_accion)
    
    if terminado:
        estado_dialogo = entorno.reiniciar(modo_entrenamiento=False)
        
    return {
        "respuesta": respuesta,
        "detalles": f"🧠 NLU: {analisis['intencion']} | Acción RL: {nombre_accion}"
    }