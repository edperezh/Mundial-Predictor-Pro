# Cambios V18 - Backtesting y fuente canónica Mundial 2026

## 1. Backtesting mejorado

Antes el backtesting podía evaluar partidos por ventana de fecha, lo que podía incluir partidos que no eran estrictamente del Mundial.

Ahora:
- reconstruye una base histórica independiente desde 2010;
- filtra estrictamente `tournament` que contenga `World Cup`;
- conserva la columna `tournament` en las features para poder filtrar correctamente;
- selecciona el mejor modelo con validación temporal antes de probar Mundial 2018 y Mundial 2022.

Nuevas columnas:
- `mejor_modelo_bt`: modelo elegido para ese backtesting.
- `nota`: explica cómo se hizo la prueba.

## 2. Partidos jugados Mundial 2026 unificados

Se agregó una fuente canónica para partidos jugados del Mundial 2026:

`get_worldcup_2026_results()`

Ahora combina:
- calendario interno actualizado,
- resultados cargados en la sesión,
- deduplicación por fecha + pareja de equipos.

Esto evita que el modelo, mapa, campeón, precisión y Monte Carlo usen listas distintas de partidos jugados.

## 3. Dedupe mejorado

Se agregó:

`dedupe_results_by_match()`

Antes se deduplicaba por fecha/equipos/marcador, por lo que una corrección de marcador podía dejar duplicados. Ahora se deduplica por:
- fecha
- pareja de equipos

y conserva la última versión.
