# Cambios V12 - Actualización real del Mundial

Problema detectado:
- En V11, si ya existía `datos/calendario_mundial_2026.csv`, la app no actualizaba partidos que antes estaban como Scheduled.
- Por eso la pestaña Campeón Mundial podía mostrar datos atrasados, por ejemplo Alemania con 1 PJ y partidos del 19 de junio sin sumar.

Mejoras:
- Se actualiza el calendario interno con los resultados del 19 de junio:
  - United States 2-0 Australia
  - Scotland 0-1 Morocco
  - Brazil 3-0 Haiti
  - Turkey 0-1 Paraguay
- Nuevo sincronizador `sync_internal_wc_calendar()`:
  - si el CSV viejo existe, lo actualiza con resultados confirmados incluidos en el código.
  - conserva `venue_key` que el usuario haya completado.
- Nueva función `fetch_free_worldcup_results_range()`:
  - intenta consultar marcadores públicos gratuitos por rango de fechas.
  - si encuentra resultados, actualiza el calendario interno.
- Nueva función `update_internal_calendar_with_results()`:
  - actualiza el calendario interno con resultados confirmados de fuentes gratuitas o CSV manual.
- La carga de datos ahora hace:
  1. sincroniza calendario interno,
  2. intenta fuentes públicas gratuitas,
  3. carga calendario interno actualizado,
  4. carga CSV manual,
  5. intenta API-Football solo si está disponible.
- API-Football sigue siendo opcional, no necesaria.

Recomendación:
- Reemplaza `app_mundial.py` y también copia `datos/calendario_mundial_2026.csv`.
- Si ya tenías un CSV viejo, la app debería corregirlo automáticamente al ejecutar.
