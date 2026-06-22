# Cambios V11 - Modo económico actualizado

Esta versión evita depender de API-Football paga para el Mundial 2026.

## Mejoras principales

- Calendario interno gratuito del Mundial 2026 en `datos/calendario_mundial_2026.csv`.
- Detección automática del partido por Equipo A vs Equipo B, sin necesitar Fixture ID pago.
- Fixture ID opcional se rellena desde el calendario interno cuando existe.
- Si API-Football bloquea la temporada 2026 por plan free, la app sigue funcionando.
- Actualización gratuita opcional de marcadores públicos con ESPN scoreboard si está disponible.
- Clima actualizado con Open-Meteo por sede seleccionada.
- Noticias/cohesión con GDELT.
- Diagnóstico de fuentes: muestra calendario interno, API-Football, clima, noticias y alineaciones.
- La fecha del partido se autocompleta con el calendario interno.
- Si no hay sede exacta, la app avisa y permite elegir sede manualmente.
- Se mantiene Monte Carlo, modelo de goles, calibración, Dixon-Coles e índice de calidad de datos.

## Importante

La API-Football gratis puede bloquear `season=2026`. Esta versión no se detiene por eso.
Para sedes exactas, completa la columna `venue_key` en `datos/calendario_mundial_2026.csv` usando una de las sedes existentes en la app.
