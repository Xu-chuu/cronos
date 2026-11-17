import json
import os
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline


class ComprensionLenguaje:
    """
    Módulo avanzado de PNL:
    1. Clasifica la intención del usuario (saludo, ánimo, charla, pregunta, profundizar, despedida).
    2. Busca conocimiento relevante usando TF-IDF + Similitud del Coseno.
    3. Mantiene memoria del tema actual de conversación.
    """

    def __init__(
        self,
        ruta_intenciones="datos/intenciones.json",
        ruta_conocimiento="datos/conocimiento.json"
    ):
        self.ruta_intenciones = ruta_intenciones
        self.ruta_conocimiento = ruta_conocimiento

        self.clasificador = make_pipeline(TfidfVectorizer(), MultinomialNB())
        self.vectorizador_conocimiento = TfidfVectorizer()
        self.temas = []
        self.matriz_conocimiento = None
        self.tema_actual = None

        self._cargar_intenciones()
        self._cargar_conocimiento()

    def _limpiar_texto(self, texto):
        texto = texto.lower().strip()
        texto = "".join(
            c for c in unicodedata.normalize("NFD", texto)
            if unicodedata.category(c) != "Mn"
        )
        texto = re.sub(r"[^a-z0-9\s]", "", texto)
        # Reduce letras repetidas excesivas (ej. "holaaa" -> "hola")
        texto = re.sub(r"(.)\1{2,}", r"\1", texto)
        return texto

    def _cargar_intenciones(self):
        if not os.path.exists(self.ruta_intenciones):
            raise FileNotFoundError(f"No se encontró: {self.ruta_intenciones}")

        with open(self.ruta_intenciones, "r", encoding="utf-8") as f:
            datos = json.load(f)

        textos, etiquetas = [], []
        for item in datos["intents"]:
            for patron in item["patterns"]:
                textos.append(self._limpiar_texto(patron))
                etiquetas.append(item["tag"])

        self.clasificador.fit(textos, etiquetas)

    def _cargar_conocimiento(self):
        if not os.path.exists(self.ruta_conocimiento):
            raise FileNotFoundError(f"No se encontró: {self.ruta_conocimiento}")

        with open(self.ruta_conocimiento, "r", encoding="utf-8") as f:
            datos = json.load(f)

        self.temas = datos["temas"]
        corpus = [
            self._limpiar_texto(t["palabras_clave"] + " " + t["resumen"])
            for t in self.temas
        ]
        self.matriz_conocimiento = self.vectorizador_conocimiento.fit_transform(corpus)

    def buscar_tema(self, texto_limpio):
        """Encuentra el tema más cercano usando Similitud del Coseno."""
        vector_usuario = self.vectorizador_conocimiento.transform([texto_limpio])
        similitudes = cosine_similarity(vector_usuario, self.matriz_conocimiento)[0]
        mejor_indice = similitudes.argmax()
        mejor_score = float(similitudes[mejor_indice])

        if mejor_score >= 0.12:
            self.tema_actual = self.temas[mejor_indice]
            return self.tema_actual, round(mejor_score, 3)

        return None, 0.0

    def analizar_mensaje(self, mensaje_usuario):
        texto_limpio = self._limpiar_texto(mensaje_usuario)
        if not texto_limpio:
            return {"intencion": "charla_cotidiana", "tema": self.tema_actual, "similitud": 0.0, "es_ambigua": False}

        # 1. Predecir intención
        probs = self.clasificador.predict_proba([texto_limpio])[0]
        idx_max = probs.argmax()
        confianza = float(probs[idx_max])
        intencion = self.clasificador.classes_[idx_max]

        # 2. Buscar si menciona algún tema de la base de conocimiento
        tema_encontrado, similitud = self.buscar_tema(texto_limpio)

        # Si detecta un tema claro (ej. "segunda guerra mundial"), prioriza 'pregunta_tema'
        if tema_encontrado and intencion not in ["despedida", "profundizar"]:
            intencion = "pregunta_tema"
        elif confianza < 0.25 and not tema_encontrado:
            intencion = "charla_cotidiana"

        return {
            "intencion": intencion,
            "confianza": round(confianza, 3),
            "tema": self.tema_actual,
            "similitud": similitud,
            "es_ambigua": len(texto_limpio.split()) <= 2 and intencion == "pregunta_tema" and not tema_encontrado
        }