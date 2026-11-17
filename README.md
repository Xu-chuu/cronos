#  Cronos — Agente Conversacional con Aprendizaje por Refuerzo y PNL

**Proyecto Académico — Inteligencia Artificial (Grupo # 18)**  
Sistema conversacional híbrido en Python que combina **Aprendizaje por Refuerzo (Q-Learning)** para la gestión estratégica del diálogo y **Procesamiento del Lenguaje Natural (PNL)** junto con **Groq (Llama 3.3)** para la generación de respuestas fluidas y naturales en tiempo real.

---

##  Arquitectura del Sistema

El agente no depende de reglas estáticas (`if-else`), sino que modela la conversación como un **Proceso de Decisión de Markov (MDP)** donde aprende mediante prueba y error a sostener diálogos coherentes y empáticos:

1. **Comprensión del Lenguaje Natural (`pnl/comprension.py`):**
   * Clasificación de intenciones del usuario mediante vectorización **TF-IDF** y **Naive Bayes Multinomial** (`scikit-learn`).
   * Recuperación semántica de contexto mediante **Similitud del Coseno** sobre una base de conocimiento multi-tema.
   * Normalización de texto y detección de ambigüedad.

2. **Entorno de Diálogo y Simulación (`entorno/`):**
   * Modela el estado de la conversación (`tiene_tema_activo`, `ultima_intencion`, `es_ambigua`).
   * Define 8 acciones conversacionales estratégicas (`saludar_amigable`, `empatizar_positivo`, `empatizar_negativo`, `conversar_cotidiano`, `responder_tema`, `profundizar_tema`, `pedir_contexto_natural`, `despedida_calida`).
   * Incluye un simulador de usuarios humanos (`simulador_usuario.py`) para entrenar al agente en miles de escenarios en segundos.

3. **Agente de Aprendizaje por Refuerzo (`agente/q_learning.py`):**
   * Implementa **Q-Learning tabular** con actualización mediante la ecuación de Bellman y política de exploración decreciente ($\epsilon$-greedy).
   * Persiste la política óptima aprendida en `modelos/tabla_q.json`.

4. **Generación de Respuestas Naturales (`pnl/generador_groq.py`):**
   * Traduce la decisión táctica tomada por el agente Q-Learning en respuestas naturales, cercanas y contextuales en tiempo real.

---

##  Estructura del Proyecto

```text
cronos/
├── agente/
│   ├── __init__.py
│   └── q_learning.py          # Algoritmo Q-Learning tabular y persistencia
├── datos/
│   ├── conocimiento.json      # Base de conocimiento temático
│   ├── intenciones.json       # Patrones de entrenamiento para PNL
│   └── respuestas.json        # Plantillas base de respaldo
├── entorno/
│   ├── __init__.py
│   ├── entorno_dialogo.py     # Definición del MDP, estados y recompensas
│   └── simulador_usuario.py   # Simulador de conversaciones para entrenamiento
├── modelos/
│   ├── tabla_q.json           # Política Q entrenada
│   └── curva_aprendizaje.png  # Gráfica de evolución de recompensas y éxito
├── pnl/
│   ├── __init__.py
│   ├── comprension.py         # Clasificador TF-IDF + Similitud del Coseno
│   └── generador_groq.py      # Generador de respuestas guiado por RL
├── utilidades/
│   ├── __init__.py
│   └── metricas.py            # Generación de gráficas de aprendizaje
├── chatear.py                 # Interfaz interactiva de conversación en consola
├── entrenar.py                # Script de entrenamiento por episodios
├── .env                       # Variables de entorno (NO subir a GitHub)
├── .gitignore                 # Exclusión de archivos sensibles y temporales
└── README.md                  # Documentación del proyecto