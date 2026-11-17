import os
import numpy as np
import matplotlib.pyplot as plt


def graficar_entrenamiento(recompensas, exitos, ventana=100, ruta_salida="modelos/curva_aprendizaje.png"):
    """
    Genera una imagen con dos gráficas:
    1. Recompensa promedio móvil por episodio.
    2. Tasa de éxito (%) en completar diálogos coherentes.
    """
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)

    episodios = np.arange(1, len(recompensas) + 1)
    media_recompensas = np.convolve(recompensas, np.ones(ventana) / ventana, mode="valid")
    media_exitos = np.convolve(exitos, np.ones(ventana) / ventana, mode="valid") * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Gráfica 1: Recompensa acumulada
    ax1.plot(episodios[ventana - 1:], media_recompensas, color="#1f77b4", linewidth=2)
    ax1.set_title("Evolución del Aprendizaje por Refuerzo (Q-Learning)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Recompensa Promedio")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Gráfica 2: Tasa de éxito
    ax2.plot(episodios[ventana - 1:], media_exitos, color="#2ca02c", linewidth=2)
    ax2.set_xlabel("Episodios de Entrenamiento")
    ax2.set_ylabel("Tasa de Éxito (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"Gráfica de aprendizaje guardada en: {ruta_salida}")