import argparse
from entorno.entorno_dialogo import EntornoDialogo
from agente.q_learning import AgenteQLearning
from utilidades.metricas import graficar_entrenamiento


def entrenar(episodios=5000):
    entorno = EntornoDialogo(max_turnos=15)
    agente = AgenteQLearning(num_acciones=len(entorno.ACCIONES))

    historial_recompensas = []
    historial_exitos = []

    print(f"Iniciando entrenamiento por {episodios} episodios...\n")

    for episodio in range(1, episodios + 1):
        estado = entorno.reiniciar(modo_entrenamiento=True)
        recompensa_total = 0.0
        terminado = False
        exito_episodio = 0

        while not terminado:
            accion = agente.elegir_accion(estado, en_entrenamiento=True)
            siguiente_estado, recompensa, terminado, info = entorno.paso(accion)

            agente.aprender(estado, accion, recompensa, siguiente_estado, terminado)

            estado = siguiente_estado
            recompensa_total += recompensa
            if info.get("exito"):
                exito_episodio = 1

        agente.reducir_epsilon()
        historial_recompensas.append(recompensa_total)
        historial_exitos.append(exito_episodio)

        if episodio % 500 == 0:
            ultimos_500_rec = sum(historial_recompensas[-500:]) / 500
            ultimos_500_exito = (sum(historial_exitos[-500:]) / 500) * 100
            print(
                f"Episodio {episodio:4d}/{episodios} | "
                f"Recompensa media: {ultimos_500_rec:6.2f} | "
                f"Tasa de éxito: {ultimos_500_exito:5.1f}% | "
                f"Epsilon: {agente.epsilon:.3f}"
            )

    agente.guardar_modelo("modelos/tabla_q.json")
    print(f"\nEntrenamiento finalizado. Estados aprendidos en tabla Q: {len(agente.tabla_q)}")
    graficar_entrenamiento(historial_recompensas, historial_exitos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar agente conversacional con Q-Learning")
    parser.add_argument("--episodios", type=int, default=5000, help="Número de episodios de entrenamiento")
    args = parser.parse_args()

    entrenar(episodios=args.episodios)