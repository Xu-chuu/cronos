import json
import os
from datetime import datetime

ARCHIVO_HISTORIAL = "datos/historial_conversacion.json"

def _cargar():
    if not os.path.exists(ARCHIVO_HISTORIAL):
        return {"sesiones": []}
    try:
        with open(ARCHIVO_HISTORIAL, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"sesiones": []}

def _guardar(datos):
    with open(ARCHIVO_HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

def guardar_mensaje(rol, texto, id_sesion="por_defecto"):
    """Guarda un mensaje en el historial. rol = 'usuario' o 'cronos'."""
    datos = _cargar()
    
    # Buscar la sesión o crearla
    sesion = next((s for s in datos["sesiones"] if s["id"] == id_sesion), None)
    if not sesion:
        sesion = {
            "id": id_sesion,
            "fecha_inicio": datetime.now().isoformat(),
            "mensajes": []
        }
        datos["sesiones"].append(sesion)
    
    sesion["mensajes"].append({
        "rol": rol,
        "texto": texto,
        "hora": datetime.now().isoformat()
    })
    
    _guardar(datos)

def obtener_historial(id_sesion="por_defecto", ultimos=20):
    """Devuelve los últimos N mensajes de una sesión."""
    datos = _cargar()
    sesion = next((s for s in datos["sesiones"] if s["id"] == id_sesion), None)
    if not sesion:
        return []
    return sesion["mensajes"][-ultimos:]

def obtener_todas_las_sesiones():
    """Devuelve un resumen de todas las sesiones."""
    datos = _cargar()
    return [
        {
            "id": s["id"],
            "fecha_inicio": s["fecha_inicio"],
            "total_mensajes": len(s["mensajes"])
        }
        for s in datos["sesiones"]
    ]

def limpiar_historial(id_sesion="por_defecto"):
    """Borra el historial de una sesión."""
    datos = _cargar()
    datos["sesiones"] = [s for s in datos["sesiones"] if s["id"] != id_sesion]
    _guardar(datos)