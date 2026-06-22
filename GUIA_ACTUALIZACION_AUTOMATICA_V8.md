# Guía V8 - Actualización automática desde internet

La versión V8 agrega actualización automática para variables avanzadas:

## Clima
Usa Open-Meteo con la sede seleccionada:
- temperatura
- humedad
- altitud aproximada de la sede

## Noticias y cohesión
Usa GDELT para noticias recientes de cada equipo.
El análisis convierte titulares recientes en:
- cohesión
- polémica/conflicto interno

Es un análisis por palabras clave. Debe interpretarse con cuidado.

## Alineaciones y lesiones
Usa API-Football si tienes API key:
- fixtures del Mundial
- lineups/alignaciones cuando estén disponibles
- injuries/lesiones cuando estén disponibles

Nota: las alineaciones suelen aparecer cerca del partido y dependen de la cobertura de la competición.

## Ratings
Si API-Football entrega alineación, la app intenta cruzar nombres con datos/jugadores_rating.csv.
Si no hay rating de un jugador, usa un valor base para no romper el cálculo.
