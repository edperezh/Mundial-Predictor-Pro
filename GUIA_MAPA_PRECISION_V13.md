# Guía V13 - Mapa del Mundial y Precisión

## Mapa del Mundial

En la pestaña "Mapa del Mundial" verás:

- Tarjetas de cada grupo.
- Posiciones, puntos, diferencia de gol y partidos jugados.
- Favoritos actuales.
- Eliminatoria proyectada aproximada.

Puedes pasar el puntero sobre cada equipo para ver:
- puntos,
- partidos jugados,
- diferencia de gol,
- estado del equipo.

## Precisión acumulada

En la pestaña "Precisión" verás:
- cuántos partidos ya se evaluaron,
- porcentaje de acierto en ganador/empate/perdedor,
- marcador exacto top 1,
- marcador exacto dentro del top 3.

Regla usada:
Si el modelo daba como resultado más probable que ganaba Equipo A, y Equipo A ganó, cuenta como acierto.
Si el modelo daba empate como resultado más probable y el partido empató, cuenta como acierto.
Si no coincide, cuenta como fallo.

## Recomendación

Cada vez que actualices resultados del Mundial:
1. Marca "Forzar actualización web".
2. Ejecuta análisis.
3. Revisa "Mapa del Mundial".
4. Revisa "Precisión".
