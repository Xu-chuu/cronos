import json
import os
import random
import numpy as np


class AgenteQLearning:
    """
    Agente de Aprendizaje por Refuerzo basado en Q-Learning tabular.
    """

    def __init__(
        self,
        num_acciones,
        tasa_aprendizaje=0.15,
        factor_descuento=0.95,
        epsilon_inicial=1.0,
        epsilon_minimo=0.05,
        decaimiento_epsilon=0.999
    ):
        self.num_acciones = num_acciones
        self.alpha = tasa_aprendizaje
        self.gamma = factor_descuento
        self.epsilon = epsilon_inicial
        self.epsilon_min = epsilon_minimo
        self.decaimiento = decaimiento_epsilon
        self.tabla_q = {}

    def _clave_estado(self, estado):
        """Convierte la tupla de estado en una cadena para usarla como clave."""
        return str(estado)

    def _asegurar_estado(self, estado):
        """Inicializa en ceros los valores Q de un estado si aún no fue visitado."""
        clave = self._clave_estado(estado)
        if clave not in self.tabla_q:
            self.tabla_q[clave] = [0.0] * self.num_acciones
        return clave

    def elegir_accion(self, estado, en_entrenamiento=True):
        """
        Selecciona una acción usando política epsilon-greedy durante el entrenamiento,
        o la mejor acción aprendida (explotación pura) durante la inferencia/chat.
        """
        clave = self._asegurar_estado(estado)

        if en_entrenamiento and random.random() < self.epsilon:
            return random.randint(0, self.num_acciones - 1)

        valores_q = self.tabla_q[clave]
        max_valor = max(valores_q)
        # Si hay empate entre acciones óptimas, elige una al azar entre ellas
        mejores_acciones = [i for i, v in enumerate(valores_q) if v == max_valor]
        return random.choice(mejores_acciones)

    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        """Actualiza el valor Q(s, a) aplicando la ecuación de Bellman."""
        clave_actual = self._asegurar_estado(estado)
        clave_siguiente = self._asegurar_estado(siguiente_estado)

        q_actual = self.tabla_q[clave_actual][accion]
        max_q_siguiente = 0.0 if terminado else max(self.tabla_q[clave_siguiente])

        nuevo_q = q_actual + self.alpha * (recompensa + self.gamma * max_q_siguiente - q_actual)
        self.tabla_q[clave_actual][accion] = round(nuevo_q, 4)

    def reducir_epsilon(self):
        """Reduce gradualmente la tasa de exploración tras cada episodio."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decaimiento)

    def guardar_modelo(self, ruta="modelos/tabla_q.json"):
        """Guarda la tabla Q aprendida en un archivo JSON."""
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as archivo:
            json.dump(self.tabla_q, archivo, indent=2, ensure_ascii=False)

    def cargar_modelo(self, ruta="modelos/tabla_q.json"):
        """Carga una tabla Q previamente entrenada desde un archivo JSON."""
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"No se encontró el modelo en '{ruta}'. Ejecuta primero 'python entrenar.py'."
            )
        with open(ruta, "r", encoding="utf-8") as archivo:
            self.tabla_q = json.load(archivo)