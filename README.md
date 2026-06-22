
# Predictor Mundial 2026 — Poisson + Machine Learning

Proyecto educativo en Python para predecir probabilidades de resultados y marcadores probables de partidos de selecciones.

## Qué hace

- Descarga resultados históricos de selecciones.
- Permite actualizar resultados del Mundial 2026 por API-Football si tienes API key.
- También permite alimentar datos manualmente con `datos/mundial_2026_manual.csv`.
- Entrena varios modelos:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - HistGradientBoosting
  - Red neuronal simple
  - XGBoost, si lo tienes instalado
- Compara modelos con:
  - accuracy
  - F1 weighted
  - log loss
  - matriz de confusión
- Genera:
  - probabilidades victoria/empate/derrota
  - goles esperados
  - top de marcadores probables
  - matriz de correlación
  - peso de variables
  - gráfica de rendimiento entre equipos

## Instalación

Abre Anaconda Prompt o terminal:

```bash
pip install -r requirements.txt
```

## Ejecutar con interfaz bonita

```bash
streamlit run app_mundial.py
```

## Ejecutar en Spyder modo consola

Abre `app_mundial.py` en Spyder y dale Run.

## API key opcional

Para actualizar Mundial 2026 desde internet:

1. Crea una cuenta en API-Football / API-Sports.
2. Copia tu API key.
3. Puedes pegarla en la barra lateral de Streamlit o crear variable de entorno:

Windows PowerShell:

```powershell
setx API_FOOTBALL_KEY "TU_API_KEY"
```

Luego cierras y abres la terminal.

## Datos manuales si no usas API

Edita:

```text
datos/mundial_2026_manual.csv
```

Columnas obligatorias:

```text
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
```

Ejemplo:

```csv
date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2026-06-17,Uzbekistan,Colombia,1,3,FIFA World Cup,Example City,United States,TRUE
```

## Nota importante

Esto no garantiza resultados exactos. El fútbol tiene ruido, tarjetas, lesiones, rotaciones, errores arbitrales y delanteros que a veces definen como si el botón de disparo estuviera dañado.
