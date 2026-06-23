# Cambios V28 - Precisión desde Túnez vs Japón

## Objetivo

Corregir la pestaña de Precisión para que el acumulado empiece exactamente desde el partido:

**Túnez vs Japón**

El usuario empezó a usar esta sección desde ese partido, por eso los encuentros anteriores no deben contar.

## Problema corregido

Antes se filtraba principalmente por fecha. Eso podía incluir partidos anteriores que aparecían con la misma fecha por diferencias de zona horaria o por datos públicos en UTC.

Ejemplo: un partido anterior podía quedar registrado como `2026-06-21` y entrar al acumulado aunque se hubiese jugado antes de Túnez vs Japón.

## Solución

Se agregaron:

- `precision_pair_key`
- `get_precision_allowed_match_keys`

Ahora la app usa el **orden real del calendario interno** y no solo la fecha.

La precisión se calcula desde el fixture:

`66456974 - Túnez vs Japón`

hacia adelante.

## Qué cambia en la app

En la pestaña Precisión ahora aparece un mensaje indicando:

- desde qué partido empieza,
- cuántos partidos anteriores fueron excluidos,
- que se usa el orden oficial del calendario para evitar errores por zona horaria.

## Archivos a subir

- app_mundial.py
- CAMBIOS_V28.md

## Comando

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Ajusta precision desde Tunez vs Japon V28"
git push
```

Luego en Streamlit Cloud:

`Manage app → Reboot app`
