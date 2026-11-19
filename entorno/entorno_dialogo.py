from entorno.simulador_usuario import SimuladorUsuario


class EntornoDialogo:
    """
    Entorno MDP enfocado en mantener conversaciones naturales, empáticas y coherentes.
    """

    ACCIONES = [
        "saludar_amigable",
        "empatizar_positivo",
        "empatizar_negativo",
        "conversar_cotidiano",
        "responder_tema",
        "profundizar_tema",
        "pedir_contexto_natural",
        "despedida_calida"
    ]

    def __init__(self, max_turnos=20):
        self.max_turnos = max_turnos
        self.simulador = SimuladorUsuario()
        self.reiniciar()

    def reiniciar(self, modo_entrenamiento=True):
        self.modo_entrenamiento = modo_entrenamiento
        self.turnos = 0
        self.terminado = False

        self.tiene_tema_activo = False
        self.ultima_intencion = "ninguna"
        self.es_ambigua = False
        # --- NUEVO: Memoria de la última acción del agente ---
        self.ultima_accion_agente = None

        if self.modo_entrenamiento:
            self.simulador.reiniciar()
            entrada = self.simulador.responder(None)
            self._actualizar_estado(entrada)

        return self.obtener_estado()

    def obtener_estado(self):
        # --- CORRECCIÓN: Usamos string en vez de tupla para que la tabla Q sea más legible ---
        tiene_tema = "con_tema" if self.tiene_tema_activo else "sin_tema"
        return f"{self.ultima_intencion}|{tiene_tema}|{self.es_ambigua}"

    def _actualizar_estado(self, analisis):
        self.ultima_intencion = analisis["intencion"]
        self.es_ambigua = analisis.get("es_ambigua", False)
        
        # --- CORRECCIÓN CRÍTICA ---
        # Actualizamos el tema activo según el análisis actual, NO acumulamos
        tema_actual = analisis.get("tema")
        if tema_actual is not None:
            self.tiene_tema_activo = True
        else:
            # Si la intención NO está relacionada con temas, reseteamos
            intenciones_sin_tema = ["saludo", "despedida", "estado_animo_positivo", 
                                    "estado_animo_negativo", "charla_cotidiana"]
            if self.ultima_intencion in intenciones_sin_tema:
                self.tiene_tema_activo = False

    def procesar_mensaje_humano(self, analisis):
        self._actualizar_estado(analisis)
        return self.obtener_estado()

    def paso(self, indice_accion):
        accion = self.ACCIONES[indice_accion]
        self.turnos += 1
        recompensa = 0.0
        exito = False

        if accion == "saludar_amigable":
            if self.ultima_intencion == "saludo":
                recompensa += 12.0
            else:
                recompensa -= 10.0

        elif accion == "empatizar_positivo":
            if self.ultima_intencion == "estado_animo_positivo":
                recompensa += 12.0
            else:
                recompensa -= 10.0

        elif accion == "empatizar_negativo":
            if self.ultima_intencion == "estado_animo_negativo":
                recompensa += 12.0
            else:
                recompensa -= 10.0

        elif accion == "conversar_cotidiano":
            if self.ultima_intencion == "charla_cotidiana":
                recompensa += 12.0
            else:
                recompensa -= 8.0

        elif accion == "responder_tema":
            if self.ultima_intencion == "pregunta_tema" and self.tiene_tema_activo and not self.es_ambigua:
                recompensa += 15.0
            else:
                recompensa -= 10.0

        elif accion == "profundizar_tema":
            if self.ultima_intencion == "profundizar" and self.tiene_tema_activo:
                recompensa += 15.0
            else:
                recompensa -= 10.0

        elif accion == "pedir_contexto_natural":
            if self.es_ambigua or (self.ultima_intencion in ["pregunta_tema", "profundizar"] and not self.tiene_tema_activo):
                recompensa += 12.0
            else:
                recompensa -= 10.0

        elif accion == "despedida_calida":
            if self.ultima_intencion == "despedida":
                recompensa += 20.0
                self.terminado = True
                exito = True
            else:
                recompensa -= 20.0
                self.terminado = True

        # --- NUEVO: Penalización por repetir la misma acción ---
        if self.ultima_accion_agente is not None and accion == self.ultima_accion_agente:
            recompensa -= 6.0  # Castigo por repetirse
        
        self.ultima_accion_agente = accion  # Guardar la acción actual para la próxima

        if not self.terminado and self.modo_entrenamiento:
            respuesta_sim = self.simulador.responder(accion)
            self._actualizar_estado(respuesta_sim)

        if self.turnos >= self.max_turnos and not self.terminado:
            self.terminado = True
            exito = True

        return self.obtener_estado(), recompensa, self.terminado, {"exito": exito, "accion": accion}