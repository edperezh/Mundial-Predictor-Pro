# Cambios V10 - Motor avanzado

Implementado:
- Simulación Monte Carlo aproximada del Mundial completo.
- Regresores de goles esperados para Equipo A y Equipo B.
- Calibración de probabilidades con Logistic Regression calibrada.
- Brier Score junto a Log Loss y F1.
- Backtesting histórico para ventanas Mundial 2018 y Mundial 2022.
- Ajuste Dixon-Coles simplificado para marcadores bajos.
- Índice de calidad de datos 0-100.
- Contexto competitivo: fase del torneo y necesidad de ganar.
- Normalización de nombres de equipos para cruzar API/histórico.
- Se mantiene actualización automática de clima, noticias, lesiones y alineaciones.

Notas:
- Monte Carlo usa bracket aproximado por siembra de fuerza, no bracket oficial FIFA.
- Variables como cohesión/noticias usan heurística por palabras clave.
- Las variables avanzadas históricas no están disponibles para todos los partidos antiguos; por eso entran como ajuste contextual y calidad de datos.
