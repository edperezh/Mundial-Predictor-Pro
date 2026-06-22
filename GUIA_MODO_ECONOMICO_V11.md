# Guía Modo Económico V11

## Objetivo

Que el predictor siga funcionando aunque no puedas pagar API-Football para temporada 2026.

## Fuentes usadas

1. Histórico internacional público.
2. `datos/calendario_mundial_2026.csv` como calendario interno.
3. `datos/mundial_2026_manual.csv` para resultados que quieras añadir/corregir.
4. Open-Meteo para clima.
5. GDELT para noticias.
6. ESPN scoreboard público como intento gratuito opcional.
7. API-Football solo como extra si tu plan lo permite.

## Cómo usar

1. Ejecuta la app.
2. Selecciona Equipo A y Equipo B.
3. La app intentará detectar el partido desde el calendario interno.
4. El Fixture ID se rellenará solo si existe en el calendario interno.
5. Si la API paga no responde, no pasa nada: el modelo sigue con fuentes gratuitas.

## Cómo actualizar resultados sin pagar

Edita:

`datos/calendario_mundial_2026.csv`

o:

`datos/mundial_2026_manual.csv`

Ejemplo para marcar un partido finalizado:

- status: Complete
- home_score: goles local
- away_score: goles visitante

## Cómo mejorar clima/sede

Si conoces la sede exacta del partido, edita `venue_key` en `datos/calendario_mundial_2026.csv`.

Debe coincidir con una de las sedes de la app, por ejemplo:

- Atlanta, United States
- Miami, United States
- Mexico City, Mexico
- Toronto, Canada
- Vancouver, Canada

Si está vacío, la app te deja elegir sede manualmente.
