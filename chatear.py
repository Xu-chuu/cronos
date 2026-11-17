import json
import random
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

from groq import Groq
from pnl.comprension import ComprensionLenguaje
from entorno.entorno_dialogo import EntornoDialogo
from agente.q_learning import AgenteQLearning


# --- CONFIGURACIÓN DEL LLM ---
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def generar_respuesta_llm(mensaje_usuario, historial, accion_agente):
    """Genera el texto usando Groq, guiado por la acción del Q-Learning."""
    
    estilos = {
        "saludar_amigable": "Saluda de forma cálida y amigable.",
        "empatizar_positivo": "Muéstrate feliz y comparte su alegría.",
        "empatizar_negativo": "Muestra empatía y apoyo.",
        "conversar_cotidiano": "Conversa de forma casual e interesante.",
        "responder_tema": "Responde con información útil sobre el tema.",
        "profundizar_tema": "Da más detalles y ejemplos.",
        "pedir_contexto_natural": "Pide más contexto de forma natural y curiosa.",
        "despedida_calida": "Despídete cordialmente."
    }
    
    system_prompt = f"""Eres Cronos, un compañero conversacional amigable.
Estilo a seguir en esta respuesta: {estilos.get(accion_agente, 'Conversa naturalmente.')}
Reglas:
- Responde SIEMPRE en español, en máximo 3 oraciones.
- Haz una pregunta de seguimiento al final.
- No repitas lo que dijo el usuario, reacciona a ello.
- Si el usuario te pregunta algo que no sabes, admítelo con naturalidad."""

    # Construir historial (últimos 6 mensajes para dar contexto)
    mensajes = [{"role": "system", "content": system_prompt}]
    for turno in historial[-6:]:
        mensajes.append(turno)
    mensajes.append({"role": "user", "content": mensaje_usuario})

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # <-- Modelo actualizado
            messages=mensajes,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Error LLM: {e}]")
        return "Interesante... cuéntame más."


def cargar_respuestas(ruta="datos/respuestas.json"):
    with open(ruta, "r", encoding="utf-8") as archivo:
        return json.load(archivo)


def construir_respuesta_natural(accion, tema, plantillas):
    """Fallback por si el LLM falla. Usa las plantillas locales."""
    opciones = plantillas.get(accion, ["¡Qué interesante! Cuéntame más sobre eso."])
    plantilla = random.choice(opciones)

    variables = ["{resumen}", "{profundizacion}", "{pregunta_seguimiento}"]
    tiene_variables = any(var in plantilla for var in variables)

    if tiene_variables and not tema:
        opciones_seguras = plantillas.get("pedir_contexto_natural", ["Cuéntame un poco más sobre eso."])
        plantilla = random.choice(opciones_seguras)

    if tema:
        plantilla = plantilla.replace("{resumen}", tema.get("resumen", ""))
        plantilla = plantilla.replace("{profundizacion}", tema.get("profundizacion", ""))
        plantilla = plantilla.replace("{pregunta_seguimiento}", tema.get("pregunta_seguimiento", ""))

    return plantilla


def iniciar_chat():
    print("=" * 60)
    print("  CRONOS - AGENTE CONVERSACIONAL CON APRENDIZAJE POR REFUERZO")
    print("  (Escribe 'salir' para terminar la conversación)")
    print("=" * 60)

    pnl = ComprensionLenguaje("datos/intenciones.json", "datos/conocimiento.json")
    entorno = EntornoDialogo(max_turnos=50)
    agente = AgenteQLearning(num_acciones=len(entorno.ACCIONES))
    agente.cargar_modelo("modelos/tabla_q.json")
    plantillas = cargar_respuestas("datos/respuestas.json")

    estado = entorno.reiniciar(modo_entrenamiento=False)
    historial = []  # Memoria real de la conversación para el LLM

    while True:
        mensaje_usuario = input("\nTú: ").strip()
        if not mensaje_usuario:
            continue
        if mensaje_usuario.lower() in ["salir", "exit", "quit"]:
            print("Cronos: ¡Un gusto charlar contigo! Nos vemos pronto.")
            break

        # 1. Comprender intención, estado de ánimo y tema
        analisis = pnl.analizar_mensaje(mensaje_usuario)
        
        # --- DEBUG (Opcional): Descomenta para ver qué extrae el NLP ---
        # print(f"[DEBUG NLP] {analisis}")
        
        estado = entorno.procesar_mensaje_humano(analisis)

        # 2. Elegir la mejor acción social aprendida por Q-Learning
        indice_accion = agente.elegir_accion(estado, en_entrenamiento=False)
        nombre_accion = entorno.ACCIONES[indice_accion]

        # 3. Actualizar estado del diálogo
        estado, recompensa, terminado, info = entorno.paso(indice_accion)

        # 4. Generar respuesta: LLM con fallback a plantillas
        try:
            respuesta = generar_respuesta_llm(mensaje_usuario, historial, nombre_accion)
        except Exception:
            respuesta = construir_respuesta_natural(nombre_accion, analisis.get("tema"), plantillas)
        
        print(f"Cronos: {respuesta}")

        # 5. Guardar en historial para que el LLM tenga memoria
        historial.append({"role": "user", "content": mensaje_usuario})
        historial.append({"role": "assistant", "content": respuesta})

        if terminado:
            print("\n--- Conversación finalizada ---")
            estado = entorno.reiniciar(modo_entrenamiento=False)
            historial = []


if __name__ == "__main__":
    iniciar_chat()