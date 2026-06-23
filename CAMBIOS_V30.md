# Cambios V30 - Precisión: top 3 de marcadores calibrado

## 1. Se elimina "Acierto marcador top 1"

La pestaña Precisión ya no muestra:

- Marcador exacto top 1
- Acierto marcador top1

La razón: en fútbol el marcador exacto #1 es una métrica demasiado estricta y poco estable. Para una app pública es más útil medir si el marcador real cae dentro del top 3.

## 2. Se mantiene y mejora "Marcador en top 3"

La métrica principal de marcadores queda:

- Marcador en top 3

## 3. Nuevo ranking calibrado de marcadores

Se agregaron funciones nuevas:

- `calibrate_score_probabilities`
- `select_top_scores_for_top3`

Estas funciones ajustan el ranking de marcadores para mejorar cobertura del top 3.

## 4. Qué corrige

La V30 corrige varios problemas:

- evita que 1-1 aparezca demasiado por defecto,
- da más peso a marcadores históricamente frecuentes como 1-0, 2-1, 2-0,
- favorece marcadores cercanos a los goles esperados,
- penaliza goleadas extremas cuando el total de goles esperado no lo justifica,
- hace que el top 3 tenga mejor cobertura entre escenarios probables.

## 5. Nueva tabla de precisión

La tabla ahora muestra:

- Top 3 marcadores
- Acierto marcador top3

Ya no muestra la columna "Acierto marcador top1".

## Cómo actualizar

Reemplaza:

- app_mundial.py
- CAMBIOS_V30.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Mejora precision top3 marcadores V30"
git push
```

En Streamlit Cloud:

- Manage app → Reboot app
