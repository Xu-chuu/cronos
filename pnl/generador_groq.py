import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GeneradorRespuestasGroq:
    """
    Genera respuestas naturales y humanas en tiempo real usando Groq,
    obedeciendo la estrategia conversacional decidida por el agente de RL.
    """

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
        self.historial = [
            {
                "role": "system",
                "content": (
                    "Eres Cronos, un compañero de conversación empático y natural. "
                    "Hablas en español neutro, como una persona real chateando por WhatsApp. "
                    "Tus respuestas deben ser concisas (2 a 4 oraciones) y fluidas. "
                    "REGLA CRÍTICA: No inventes datos, fechas, nombres de episodios ni hechos. "
                    "Si no estás seguro de algo, di que no lo recuerdas o que no lo sabes. "
                    "Si el usuario parece satisfecho con una respuesta anterior, no sigas profundizando; "
                    "cambia de tema o pregúntale qué le pareció. "
                    "REGLA DE LISTAS: Si el usuario te pide una lista, da máximo 3 elementos y di 'entre otras'. "
                    "No intentes listar todo el mundo ni des respuestas eternas."
                )
            }
        ]

    def generar(self, mensaje_usuario, accion_rl):
        guia_estrategica = self.INSTRUCCIONES_ESTRATEGIA.get(
            accion_rl,
            "Responde de forma natural y mantén viva la conversación."
        )

        self.historial.append({"role": "user", "content": mensaje_usuario})

        mensajes_envio = self.historial + [
            {
                "role": "system",
                "content": f"[Directiva del Agente RL para este turno]: {guia_estrategica}"
            }
        ]

        try:
            respuesta = self.cliente.chat.completions.create(
                model=self.modelo,
                messages=mensajes_envio,
                temperature=0.7,
                max_tokens=600  # <-- Aumentado para evitar respuestas cortadas
            )
            texto_respuesta = respuesta.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Error LLM en API: {e}]")
            texto_respuesta = "¡Qué interesante! Cuéntame un poco más sobre eso."

        self.historial.append({"role": "assistant", "content": texto_respuesta})

        # Limitar el historial para no gastar tokens de más (mantiene el primer system prompt)
        if len(self.historial) > 13:
            self.historial = [self.historial[0]] + self.historial[-12:]

        return texto_respuesta