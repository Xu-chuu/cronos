import random


class SimuladorUsuario:
    """
    Simula conversaciones humanas naturales (saludos, estados de ánimo,
    charla casual, preguntas sobre temas y despedidas).
    """

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.turnos_charla = 0
        self.meta_turnos = random.randint(3, 6)
        self.saludo_hecho = False
        self.tema_activo = False

    def responder(self, accion_agente):
        if accion_agente is None:
            self.saludo_hecho = True
            return {"intencion": "saludo", "tema": None, "es_ambigua": False}

        self.turnos_charla += 1

        if accion_agente == "saludar_amigable":
            # El usuario responde cómo está (positivo o negativo)
            animo = random.choice(["estado_animo_positivo", "estado_animo_negativo"])
            return {"intencion": animo, "tema": None, "es_ambigua": False}

        # Si ya conversó lo suficiente, el usuario se despide
        if self.turnos_charla >= self.meta_turnos:
            return {"intencion": "despedida", "tema": None, "es_ambigua": False}

        # Flujo natural en medio de la charla
        if accion_agente in ["empatizar_positivo", "empatizar_negativo", "conversar_cotidiano"]:
            if random.random() < 0.6:
                self.tema_activo = True
                return {"intencion": "pregunta_tema", "tema": {"id": "simulado"}, "es_ambigua": False}
            return {"intencion": "charla_cotidiana", "tema": None, "es_ambigua": False}

        if accion_agente == "responder_tema":
            if random.random() < 0.5:
                return {"intencion": "profundizar", "tema": {"id": "simulado"}, "es_ambigua": False}
            return {"intencion": "charla_cotidiana", "tema": None, "es_ambigua": False}

        if accion_agente == "profundizar_tema":
            return {"intencion": "charla_cotidiana", "tema": {"id": "simulado"}, "es_ambigua": False}

        if accion_agente == "pedir_contexto_natural":
            self.tema_activo = True
            return {"intencion": "pregunta_tema", "tema": {"id": "simulado"}, "es_ambigua": False}

        return {"intencion": "charla_cotidiana", "tema": None, "es_ambigua": False}