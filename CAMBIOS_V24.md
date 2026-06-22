# Cambios V24 - API fija, idiomas, estabilidad y empates

## 1. API_FOOTBALL fija y segura

La app ahora puede leer la clave directamente desde Streamlit Secrets o variables de entorno.

Configura en Streamlit Cloud:

```toml
API_FOOTBALL_KEY = "TU_CLAVE_REAL"
```

Si existe esa clave, el campo de API ya no se muestra como input público; la app muestra que la API está configurada desde Secrets.

También acepta:

- FOOTBALL_API_KEY
- API_SPORTS_KEY

## 2. Selector de idioma

Se agregó selector:

- Español
- English
- Português
- Français

Traduce las secciones principales de la interfaz: título, configuración, botones, pestañas y métricas principales.

## 3. Mejor estabilidad de predicción

Se ajustó el cálculo de estabilidad para que sea más realista:

- no castiga de más variaciones moderadas entre modelos,
- tiene en cuenta el margen entre la probabilidad más alta y la segunda,
- usa el modelo oficial estable como referencia.

## 4. Mejor manejo de empates

Se redujo la inflación automática del empate.

Antes, muchos partidos terminaban con 1-1 como marcador top incluso cuando había favorito claro. Ahora:

- el empate solo sube si el partido es realmente cerrado,
- se limita el empate cuando hay favorito fuerte,
- se reduce el sesgo automático hacia marcadores iguales,
- Dixon-Coles queda más conservador.

## 5. Marcadores más probables más equilibrados

Se agregó calibración de marcadores iguales para que 1-1 no aparezca como máximo por defecto en casi todos los partidos.

## Cómo actualizar

Reemplaza:

- app_mundial.py
- requirements.txt
- CAMBIOS_V24.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega API fija idiomas estabilidad y mejora empates V24"
git push
```

En Streamlit Cloud:

- Manage app → Reboot app
