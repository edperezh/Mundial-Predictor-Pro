# Guía V12 - Mantener el Mundial actualizado sin pagar API

## Qué hace automáticamente

Cada vez que presionas:
`🚀 Ejecutar análisis y entrenar modelos`

la app intenta:

1. Sincronizar `datos/calendario_mundial_2026.csv` con los resultados confirmados incluidos en el código.
2. Consultar fuentes públicas gratuitas por fechas.
3. Actualizar el calendario interno si encuentra marcadores nuevos.
4. Cargar `datos/mundial_2026_manual.csv` si quieres corregir/agregar resultados manualmente.
5. Usar API-Football solo si tu plan lo permite.

## Cómo verificar si está actualizado

En la pestaña Campeón Mundial revisa:
- PJ Mundial
- PTS Mundial
- DG Mundial

Después de esta versión, ya deberían aparecer partidos del 19 de junio sumados:
- United States 2-0 Australia
- Scotland 0-1 Morocco
- Brazil 3-0 Haiti
- Turkey 0-1 Paraguay

## Cómo corregir manualmente un resultado

Abre:
`datos/calendario_mundial_2026.csv`

Busca el partido y cambia:

- status = Complete
- home_score = goles local
- away_score = goles visitante

Guarda el CSV y vuelve a ejecutar la app.

## Qué pasa si una fuente pública no responde

No se rompe nada. La app usa:
- calendario interno,
- CSV manual,
- histórico,
- clima Open-Meteo,
- noticias GDELT,
- modelo propio.

## Qué archivo debes copiar

- `app_mundial.py`
- `datos/calendario_mundial_2026.csv`
- `CAMBIOS_V12.md`
- `GUIA_ACTUALIZAR_MUNDIAL_V12.md`
