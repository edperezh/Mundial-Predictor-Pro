# Cambios V21 - Mejora de precisión avanzada

Esta versión agrega las recomendaciones principales para mejorar predicciones:

## 1. Modelos de goles esperados reforzados
- Se agregó PoissonRegressor para goles.
- Los regresores de goles usan pesos por importancia del partido.
- Los goles esperados mezclan modelo heurístico + regresor ML.

## 2. Peso por tipo de partido
- Los entrenamientos usan sample_weight.
- Mundial, eliminatorias y copas oficiales pesan más.
- Amistosos pesan menos.
- Partidos recientes pesan más que partidos antiguos.

## 3. Forma reciente multi-ventana
- Últimos 3, 5, 10 y 20 partidos.
- Goles, goles recibidos, puntos, victorias, porterías en cero y tasas de anotación.

## 4. Rachas
- Racha de victorias.
- Racha invicto.
- Racha sin marcar.
- Racha de porterías en cero.
- Tendencia de empates y partidos de bajo marcador.

## 5. Mejor detección de empates
- Nuevas señales: elo_close_index, draw_signal, defensive balance, low-score tendency.
- Ajuste moderado de empate en partidos cerrados.

## 6. Ensamble ponderado
- Ya no depende solo del mejor modelo.
- Combina modelos con pesos según log_loss, brier y f1.

## 7. Jugadores por posición
- Rating arquero, defensa, mediocampo, ataque y banca.
- Ausencias clave por posición.
- Perder arquero/delantero/defensa titular pesa distinto.

## 8. Backtesting más detallado
- Agrega métricas por favoritos, empates, partidos cerrados y fase de grupos.

## Nota
Estas mejoras no garantizan más acierto en todos los escenarios, pero dan al modelo más información útil y mejor calibración.
