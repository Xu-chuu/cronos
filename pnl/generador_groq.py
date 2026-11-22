import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GeneradorRespuestasGroq:
    INSTRUCCIONES_ESTRATEGIA = {
        "saludar_amigable": "Saluda con calidez y pregunta cómo está la persona. No des información, solo saluda.",
        "empatizar_positivo": "Alégrate por la persona y comparte una emoción positiva breve. Luego pregunta algo ligero.",
        "empatizar_negativo": "Muestra empatía breve, sin sermones. Pregunta si quiere hablar de eso o cambiar de tema.",
        "conversar_cotidiano": "Reacciona brevemente a lo que dijo el usuario y elige UN solo tema para preguntar. No des discursos.",
        "responder_tema": "Responde en máximo 2 oraciones sobre el tema. No des clases ni listas. Termina con una pregunta corta.",
        "profundizar_tema": "Da UN dato curioso breve sobre el tema. Si ya diste un dato antes, mejor cambia de tema con una pregunta.",
        "pedir_contexto_natural": "Pide de forma amable que te cuente un poco más de contexto. Sé breve.",
        "despedida_calida": "Despídete con afecto y agradece la conversación."
    }

    def __init__(self, modelo="openai/gpt-oss-120b"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "TU_CLAVE_NUEVA":
            raise ValueError("No se encontró una GROQ_API_KEY válida en el archivo .env")
        
        self.cliente = Groq(api_key=api_key)
        self.modelo = modelo
        
        self.mensaje_sistema = (
            "Eres Cronos, un compañero de conversación empático y natural. "
            "Hablas en español neutro, como una persona real chateando. "
            "Tus respuestas deben ser concisas (2 a 4 oraciones) y fluidas. "
            "REGLA CRÍTICA: No inventes datos, fechas, nombres de episodios ni hechos históricos. "
            "Si no estás seguro de algo, di que no lo recuerdas. "
            "Si el usuario parece satisfecho con una respuesta anterior, no sigas profundizando; "
            "cambia de tema o pregúntale qué le pareció. "
            "REGLA DE LISTAS: Si el usuario te pide una lista, da máximo 3 elementos y di 'entre otras'."
        )
        
        self.historial = [{"role": "system", "content": self.mensaje_sistema}]

    def generar(self, mensaje_usuario, accion_rl, tema_conocido=None):
        guia_estrategica = self.INSTRUCCIONES_ESTRATEGIA.get(
            accion_rl,
            "Responde de forma natural y mantén viva la conversación."
        )

        # Verificar si ya hubo un saludo para no repetirlo
        ya_saludo = any("hola" in m["content"].lower() for m in self.historial if m["role"] == "assistant")
        if accion_rl == "saludar_amigable" and ya_saludo:
            guia_estrategica = "Ya saludaste antes. Ahora continúa la conversación de forma natural sin volver a saludar."

        self.historial.append({"role": "user", "content": mensaje_usuario})

        contexto_extra = ""
        fuente = "llm"

        if accion_rl in ["responder_tema", "profundizar_tema"]:
            if tema_conocido:
                # 🟢 VERDE: Tiene la info en el JSON
                contexto_extra = (
                    f"\n[INFORMACIÓN DE TU BASE DE CONOCIMIENTO]:\n"
                    f"Resumen: {tema_conocido.get('resumen', '')}\n"
                    f"Profundización: {tema_conocido.get('profundizacion', '')}\n"
                    f"[FIN DE LA INFORMACIÓN]\n"
                    f"Usa SOLO esta información para responder."
                )
                fuente = "conocimiento"
            else:
                # 🟡 AMARILLO
                contexto_extra = (
                    "\n[AVISO]: No tienes información en tu base de datos sobre un tema específico. "
                    "Sin embargo, el usuario acaba de hablar de algo en los mensajes anteriores. "
                    "Responde siguiendo el hilo de la conversación de forma natural."
                )
                fuente = "llm"
        else:
            # 🟡 AMARILLO: Charla casual
            contexto_extra = (
                "\n[AVISO]: Es una conversación casual. Estás chateando con un humano. "
                "REGLA DE ORO: Si el usuario dice cosas como 'ehh?', 'no entiendo' o 'eh', significa que no te expresaste bien o que estás siendo repetitivo. "
                "NUNCA repitas la misma frase que dijiste antes. Cambia el tema o haz una pregunta diferente. "
                "No hables de Inteligencia Artificial ni de aprendizaje por refuerzo a menos que te lo pidan explícitamente. "
                "Mantén la conversación fluida y natural."
            )
            fuente = "llm"

        mensajes_envio = self.historial + [
            {
                "role": "system",
                "content": f"[Directiva del Agente RL para este turno]: {guia_estrategica}{contexto_extra}"
            }
        ]

        try:
            respuesta = self.cliente.chat.completions.create(
                model=self.modelo,
                messages=mensajes_envio,
                temperature=0.7,
                max_tokens=600
            )
            texto_respuesta = respuesta.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Error LLM en API: {e}]")
            texto_respuesta = "¡Qué interesante! Cuéntame un poco más sobre eso."
            fuente = "error"

        self.historial.append({"role": "assistant", "content": texto_respuesta})

        if len(self.historial) > 13:
            self.historial = [self.historial[0]] + self.historial[-12:]

        return texto_respuesta, fuente

    def reiniciar_memoria(self):
        """Reinicia el historial del LLM cuando el usuario cambia de chat."""
        self.historial = [{"role": "system", "content": self.mensaje_sistema}]