# Cambios V23 - Modo Estable / Producción

## Objetivo

Evitar que las predicciones públicas cambien mucho cada vez que se ejecuta el programa.

## Mejoras aplicadas

### 1. Semilla global fija

Se agregó:

- `SEED = 42`
- `RANDOM_STATE = 42`
- `np.random.seed(SEED)`
- `random.seed(SEED)`
- `PYTHONHASHSEED`

Esto reduce variaciones en Random Forest, XGBoost, redes neuronales y modelos similares.

### 2. Modelo oficial fijo

La app ahora distingue entre:

- Mejor modelo en validación.
- Modelo oficial usado para predicciones públicas.

Por defecto, el modelo oficial es:

`Logistic Regression calibrada`

Esto evita que la predicción pública cambie porque en una ejecución ganó Random Forest y en otra ganó XGBoost.

### 3. Modo producción estable

Las predicciones usan:

`prediction_mode = "oficial_estable"`

El ensamble sigue existiendo como apoyo/análisis, pero la predicción pública se basa en el modelo oficial fijo.

### 4. Guardado de modelo entrenado

Se agregaron:

- `modelos/modelo_oficial.pkl`
- `modelos/metadata_modelo.json`

La app guarda el resultado del entrenamiento estable. Si existe, puede cargarse desde el botón:

`📦 Cargar guardado`

### 5. Snapshot de resultados Mundial 2026

Se agrega:

`datos/snapshot_mundial_2026.csv`

Esto congela los partidos del Mundial 2026 que está usando el modelo al momento de entrenar.

### 6. Métrica de estabilidad

La app muestra:

- estabilidad del modelo: Alta / Media / Baja
- variación promedio entre modelos

Esto ayuda a saber si una predicción es sólida o si los modelos están muy divididos.

### 7. Panel de versión del modelo

En la pestaña Modelos ahora se muestra:

- modelo oficial
- mejor modelo por validación
- versión
- fecha de entrenamiento
- rango train/test temporal
- archivo del modelo
- snapshot del Mundial

## Cómo actualizar

Reemplaza principalmente:

- app_mundial.py
- requirements.txt
- CAMBIOS_V23.md

Luego ejecuta:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega modo estable de produccion V23"
git push
```

En Streamlit Cloud:

- espera redeploy automático o
- Manage app → Reboot app

## Nota importante

La primera vez debes presionar:

`🚀 Reentrenar modelo estable`

Después podrás usar:

`📦 Cargar guardado`

si el archivo existe en el entorno.
