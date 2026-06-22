# Cambios V22 - Corrección error streak_stats y tablas Streamlit Cloud

## Error corregido 1

En Streamlit Cloud aparecía:

`AttributeError: 'TeamMemory' object has no attribute 'streak_stats'`

## Causa

Streamlit puede conservar objetos antiguos en `st.session_state` entre redeploys. Si el objeto `TeamMemory` fue creado antes de la V21, no tenía el método nuevo `streak_stats`.

## Solución

Se agregó una función segura:

`safe_streak_stats(mem, team, n)`

Ahora, si el objeto de memoria no tiene `streak_stats`, el código calcula las rachas de forma compatible sin romper la app.

## Error corregido 2

En los logs aparecía:

`Could not convert 'Fase de grupos' with type str: tried to convert to double`

## Causa

Algunas tablas de Streamlit mezclaban números y texto en una misma columna llamada `Valor`.

## Solución

Se agregó:

`safe_streamlit_df(df)`

Ahora las columnas mixtas se convierten de forma segura antes de mostrarse con `st.dataframe`.

## Archivos principales a subir

- app_mundial.py
- CAMBIOS_V22.md

## Nota

Los mensajes de `use_container_width` son advertencias de Streamlit, no rompen la app.
