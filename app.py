import streamlit as st
from pnl.comprension import ComprensionLenguaje
from pnl.generador_groq import GeneradorRespuestasGroq
from entorno.entorno_dialogo import EntornoDialogo
from agente.q_learning import AgenteQLearning

# 1. Configuración de la página web
st.set_page_config(page_title="Cronos AI - Grupo #18", page_icon="", layout="centered")

st.title("Cronos")
st.markdown("**Agente Conversacional con Aprendizaje por Refuerzo (Grupo #18)**")

# 2. Inicializar los modelos de IA en la caché de la sesión (para que no se recarguen con cada mensaje)
if "modelos_cargados" not in st.session_state:
    st.session_state.pnl = ComprensionLenguaje("datos/intenciones.json", "datos/conocimiento.json")
    st.session_state.entorno = EntornoDialogo(max_turnos=50)
    st.session_state.agente = AgenteQLearning(num_acciones=len(st.session_state.entorno.ACCIONES))
    st.session_state.agente.cargar_modelo("modelos/tabla_q.json")
    st.session_state.generador = GeneradorRespuestasGroq()
    
    st.session_state.estado_dialogo = st.session_state.entorno.reiniciar(modo_entrenamiento=False)
    st.session_state.mensajes = []
    st.session_state.modelos_cargados = True

# 3. Mostrar el historial de mensajes en la pantalla
for msj in st.session_state.mensajes:
    with st.chat_message(msj["role"]):
        st.markdown(msj["content"])
        # Mostrar métricas técnicas si existen (solo en los mensajes del bot)
        if "metricas" in msj:
            st.caption(msj["metricas"])

# 4. Capturar el mensaje del usuario
if prompt := st.chat_input("Escribe tu mensaje aquí..."):
    # Mostrar el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.mensajes.append({"role": "user", "content": prompt})

    # PROCESAMIENTO INTERNO (PLN + Q-LEARNING + GROQ)
    analisis = st.session_state.pnl.analizar_mensaje(prompt)
    st.session_state.estado_dialogo = st.session_state.entorno.procesar_mensaje_humano(analisis)
    
    indice_accion = st.session_state.agente.elegir_accion(st.session_state.estado_dialogo, en_entrenamiento=False)
    nombre_accion = st.session_state.entorno.ACCIONES[indice_accion]
    
    estado, recompensa, terminado, info = st.session_state.entorno.paso(indice_accion)
    respuesta = st.session_state.generador.generar(prompt, nombre_accion)

    # Mostrar la respuesta del agente y las métricas para la ingeniera
    with st.chat_message("assistant"):
        st.markdown(respuesta)
        info_tecnica = f" *NLU: {analisis['intencion']} | Similitud: {analisis['similitud']} | Acción RL: {nombre_accion}*"
        st.caption(info_tecnica)
    
    st.session_state.mensajes.append({
        "role": "assistant", 
        "content": respuesta, 
        "metricas": info_tecnica
    })

    if terminado:
        st.session_state.estado_dialogo = st.session_state.entorno.reiniciar(modo_entrenamiento=False)