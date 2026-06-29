# Cambios V41.1 - Corrección AutoRefresh

## Error corregido

Se corrigió el error que aparecía después de actualizar automáticamente partidos del Mundial:

```text
teams = sorted(set(results["home_team"]).union(set(results["away_team"])))
```

La causa era que una función devolvía una tupla `(dataframe, info)` y la V41 estaba guardando esa tupla completa en `st.session_state.results`.

## Solución

- Se desempaqueta correctamente `replace_worldcup_results_with_canonical`.
- Se agregó defensa antes de crear la lista de equipos.
- Si por error llega una tupla, la app recupera el DataFrame.
- Si faltan columnas `home_team` o `away_team`, muestra mensaje claro y no rompe toda la app.

## No afecta

- Dominio
- Landing web
- Google Search Console
- Vercel
- Mercado Pago

Solo reemplaza `app_mundial.py` en la app Streamlit.
