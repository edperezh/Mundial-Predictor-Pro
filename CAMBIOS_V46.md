# CAMBIOS V46 — Eliminados en 0% y consistencia operativa

## Objetivo
Corregir la probabilidad estimada de campeón para que todo equipo eliminado del Mundial quede en 0%, incluso cuando la fuente pública entregue información de desempate o penales por columnas adicionales.

## Mejoras
- Se agregó detección robusta de ganador/perdedor real desde:
  - marcador normal;
  - campos `winner_team` / `loser_team`;
  - banderas `home_winner` / `away_winner`;
  - columnas de penales si existen.
- Los partidos finalizados desde el inicio de eliminatorias eliminan automáticamente al perdedor.
- Los no clasificados a dieciseisavos quedan en 0% cuando ya hay 72 partidos de grupos finalizados.
- Respaldo opcional por Secrets: `WORLD_CUP_2026_ELIMINATED_TEAMS`.
- La tabla de campeón ahora muestra Estado: Vivo / Eliminado.
- El ranking renormaliza los vivos a 100% y mantiene eliminados en 0%.
- Se preservan columnas de ganador desde ESPN público cuando estén disponibles.

## Uso opcional
Si una fuente pública tarda en actualizar un desempate, puedes usar temporalmente:

```toml
WORLD_CUP_2026_ELIMINATED_TEAMS = "Japan, Sweden"
```

Usa esta opción solo como respaldo manual mientras la fuente pública se actualiza.

## Nota de comunicación
La app sigue siendo una herramienta educativa de análisis deportivo. Las probabilidades son estimaciones estadísticas y no garantizan resultados.
