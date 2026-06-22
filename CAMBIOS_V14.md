# Cambios V14 - Corrección error fixture_id

## Error corregido

Se corrigió el error:

`'fixture_id' is both an index level and a column label, which is ambiguous.`

## Causa

En la versión V13, durante la sincronización del calendario interno, `fixture_id` quedaba al mismo tiempo como índice y como columna de pandas. Cuando el código intentaba ordenar por `fixture_id`, pandas no sabía cuál usar.

## Solución

Antes de ordenar el calendario interno, ahora se hace:

`current = current.reset_index(drop=True)`

y luego se ordena normalmente.

## Qué conserva

- Mapa del Mundial.
- Pestaña Precisión.
- Calendario interno.
- Actualización gratuita.
- Monte Carlo.
- Modelo de goles.
- Calidad de datos.
