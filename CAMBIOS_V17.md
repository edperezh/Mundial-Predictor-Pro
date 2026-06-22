# Cambios V17 - Backtesting corregido

## Problema corregido

En versiones anteriores, el backtesting Mundial 2018 / 2022 podía salir con:

- partidos_test = 0
- accuracy = None
- log_loss = None
- brier_score = None

Esto pasaba porque el backtesting usaba la misma base filtrada por el slider principal
"Usar partidos desde el año". Si el usuario tenía 2024, no existían partidos de 2018 ni 2022
en la base evaluada.

## Solución

El backtesting ahora reconstruye una base histórica independiente desde 2010.

Por eso ya no depende del slider principal.

## Resultado esperado

Al ejecutar "Ejecutar backtesting histórico", debe mostrar partidos evaluados para:
- Mundial 2018
- Mundial 2022

y métricas:
- accuracy
- log_loss
- brier_score

## Nota

El backtesting es una prueba histórica aproximada con ventanas de fechas del Mundial.
Sirve para medir comportamiento del modelo en torneos pasados.
