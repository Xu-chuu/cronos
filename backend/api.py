from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pnl.comprension import ComprensionLenguaje
from pnl.generador_groq import GeneradorRespuestasGroq
from entorno.entorno_dialogo import EntornoDialogo
from agente.q_learning import AgenteQLearning
from utilidades.feedback import guardar_feedback
from utilidades.historial import guardar_mensaje, obtener_historial, limpiar_historial, obtener_todas_las_sesiones

app = FastAPI(title="API de Cronos")

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

# Variable global para recordar la sesión actual en el backend
ultima_sesion_usada = ""

class Mensaje(BaseModel):
    texto: str
    sesion_id: str = "chat_por_defecto"

class PeticionFeedback(BaseModel):
    mensaje: str
    respuesta: str
    accion_rl: str
    calificacion: int  # 1 = 👍, -1 = 👎

@app.post("/chatear")
def procesar_chat(mensaje: Mensaje):
    global estado_dialogo, ultima_sesion_usada
    
    # Si el usuario cambió de chat, reiniciamos la memoria del RL y cargamos el contexto
    if ultima_sesion_usada != mensaje.sesion_id:
        estado_dialogo = entorno.reiniciar(modo_entrenamiento=False)
        generador.reiniciar_memoria() 
        
        mensajes_previos = obtener_historial(id_sesion=mensaje.sesion_id, ultimos=6)
        for m in mensajes_previos:
            rol = "user" if m["rol"] == "usuario" else "assistant"
            generador.historial.append({"role": rol, "content": m["texto"]})
            
        ultima_sesion_usada = mensaje.sesion_id

    # 1. Guardar mensaje del usuario
    guardar_mensaje("usuario", mensaje.texto, mensaje.sesion_id)
    
    # 2. Comprensión (NLU)
    analisis = pnl.analizar_mensaje(mensaje.texto)
    analisis["mensaje_original"] = mensaje.texto
    estado_dialogo = entorno.procesar_mensaje_humano(analisis)
    
    # 3. Decisión (RL)
    indice_accion = agente.elegir_accion(estado_dialogo, en_entrenamiento=False)
    nombre_accion = entorno.ACCIONES[indice_accion]
    estado_dialogo, recompensa, terminado, info = entorno.paso(indice_accion)
    
    # 4. Generación (NLG) - El LLM decide todo
    tema = analisis.get("tema")
    respuesta, fuente = generador.generar(mensaje.texto, nombre_accion, tema)
    
    # 5. Guardar respuesta
    guardar_mensaje("cronos", respuesta, mensaje.sesion_id)
    
    if terminado:
        estado_dialogo = entorno.reiniciar(modo_entrenamiento=False)
        
    return {
        "respuesta": respuesta,
        "detalles": f"🧠 NLU: {analisis['intencion']} | Acción RL: {nombre_accion}",
        "fuente": fuente,
        "accion_rl": nombre_accion
    }

# --- ENDPOINT DE FEEDBACK ---
@app.post("/feedback")
def recibir_feedback(datos: PeticionFeedback):
    """Guarda el feedback y actualiza el agente EN VIVO."""
    guardar_feedback(datos.mensaje, datos.respuesta, datos.accion_rl, datos.calificacion)
    
    try:
        indice_accion = entorno.ACCIONES.index(datos.accion_rl)
        
        if datos.calificacion == -1:
            agente.aprender(estado_dialogo, indice_accion, -15.0, estado_dialogo, False)
            print(f"👎 Castigo aplicado a la acción: {datos.accion_rl}")
        elif datos.calificacion == 1:
            agente.aprender(estado_dialogo, indice_accion, 10.0, estado_dialogo, False)
            print(f"👍 Premio aplicado a la acción: {datos.accion_rl}")
            
        agente.guardar_modelo("modelos/tabla_q.json")
        
    except ValueError:
        print(f"Error: La acción '{datos.accion_rl}' no es válida.")
    
    return {"estado": "ok", "mensaje": "Feedback guardado y agente actualizado"}

# --- ENDPOINTS PARA LAS SESIONES ---
@app.get("/sesiones")
def listar_sesiones():
    return {"sesiones": obtener_todas_las_sesiones()}

@app.get("/historial/{sesion_id}")
def ver_historial_sesion(sesion_id: str, ultimos: int = 50):
    return {"mensajes": obtener_historial(id_sesion=sesion_id, ultimos=ultimos)}

@app.delete("/historial/{sesion_id}")
def borrar_historial_sesion(sesion_id: str):
    limpiar_historial(id_sesion=sesion_id)
    return {"estado": "ok", "mensaje": "Chat borrado"}