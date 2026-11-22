import json

def guardar_feedback(mensaje, respuesta, accion_rl, calificacion):
    """Guarda la calificación (1 = buena, -1 = mala) en el archivo."""
    try:
        with open("datos/feedback.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"feedback": []}
    
    data["feedback"].append({
        "mensaje": mensaje,
        "respuesta": respuesta,
        "accion_rl": accion_rl,
        "calificacion": calificacion  # 1 = 👍, -1 = 👎
    })
    
    with open("datos/feedback.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def obtener_recompensa_extra(mensaje, accion_rl):
    """Busca si esta combinación ya fue mal calificada antes."""
    try:
        with open("datos/feedback.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return 0.0
    
    for item in data["feedback"]:
        if item["mensaje"] == mensaje and item["accion_rl"] == accion_rl:
            if item["calificacion"] == -1:
                return -15.0  # Penalización fuerte por respuesta mal calificada
            elif item["calificacion"] == 1:
                return +10.0  # Bonus por respuesta bien calificada
    return 0.0