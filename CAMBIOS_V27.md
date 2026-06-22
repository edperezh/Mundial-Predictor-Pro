# Cambios V27 - Conteo automático real sin modificar Secrets

## Objetivo

Evitar tener que cambiar manualmente `WORLD_CUP_2026_PLAYED_COUNT` en Streamlit Secrets.

Ahora la app intenta contar automáticamente los partidos del Mundial 2026 usando una fuente pública gratuita.

## 1. Conteo automático desde fuente pública

Se agregó:

- `espn_scoreboard_to_events`
- `fetch_free_worldcup_events_range`
- `automatic_worldcup_public_count`
- `replace_worldcup_results_with_canonical`

La app consulta marcadores públicos por fecha desde el inicio del Mundial hasta el día actual.

## 2. Diferencia entre finalizados y en vivo

La app distingue:

- partidos finalizados: sí alimentan el modelo,
- partidos en vivo: cuentan como iniciados/jugados en pantalla, pero NO se usan para entrenar hasta que terminen,
- partidos programados: no se usan como jugados.

Esto evita que el modelo entrene con resultados incompletos.

## 3. Todo el programa usa la misma fuente canónica

Antes podía pasar que:

- el mapa usara unos partidos,
- precisión usara otros,
- campeón usara otro conteo,
- entrenamiento tuviera duplicados.

Ahora, antes de entrenar, la app reemplaza los resultados del Mundial 2026 por una fuente canónica automática.

Eso alinea:

- entrenamiento,
- mapa,
- precisión,
- campeón,
- snapshot,
- Monte Carlo.

## 4. Sin Secrets manual para conteo

Ya no necesitas modificar:

```toml
WORLD_CUP_2026_PLAYED_COUNT = "42"
```

Ese valor queda solo como respaldo opcional si una fuente pública no responde.

## 5. Panel visible más claro

La app ahora puede mostrar:

```text
Mundial 2026: 41 partidos finalizados + 1 en vivo
Iniciados/jugados: 42
Fuente: ESPN público automático
```

## 6. Caché inteligente

Para fechas recientes, la app no usa caché viejo. Eso ayuda a actualizar partidos del día.

## Cómo actualizar

Reemplaza:

- app_mundial.py
- requirements.txt
- CAMBIOS_V27.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Automatiza conteo real del Mundial V27"
git push
```

En Streamlit Cloud:

- Manage app → Reboot app

## Nota

Si la fuente pública gratuita cambia o bloquea requests, la app usa fallback con calendario interno/manual/API. La clave API_FOOTBALL sigue funcionando para enriquecer datos si tienes cobertura.
