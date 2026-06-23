# Cambios V31 - Estadio automático para clima y altitud

## Objetivo

Mejorar la opción `Sede/estadio para clima y altitud` para que la app intente detectar automáticamente el estadio/lugar real del partido seleccionado.

## Qué se agregó

### 1. Base de estadios con nombre real

La base de sedes ahora incluye estadio, ciudad, país, latitud, longitud y altitud:

- Mercedes-Benz Stadium
- Gillette Stadium
- AT&T Stadium
- NRG Stadium
- GEHA Field at Arrowhead Stadium
- SoFi Stadium
- Hard Rock Stadium
- MetLife Stadium
- Lincoln Financial Field
- Levi's Stadium
- Lumen Field
- Estadio Akron
- Estadio Azteca
- Estadio BBVA
- BMO Field
- BC Place

## 2. Resolución automática del estadio

Nueva función:

```python
resolve_worldcup_match_venue(...)
```

Jerarquía profesional:

1. Calendario interno si tiene `venue_key` exacto.
2. API-Football si devuelve venue name/city.
3. ESPN público si devuelve venue.
4. Sede seleccionada por usuario.
5. Referencia profesional por país o sede neutral.

## 3. Selector más claro

El selector ahora muestra:

```text
Estadio · Ciudad, País
```

Ejemplo:

```text
Mercedes-Benz Stadium · Atlanta, United States
```

## 4. Se adapta al versus elegido

Cuando el usuario cambia Equipo A / Equipo B, la app vuelve a sugerir la sede correspondiente.

Si no encuentra partido oficial o estadio exacto, usa una sede base razonable y avisa la fuente/confianza.

## 5. Diagnóstico de sede

La tabla de variables ahora muestra:

- estadio/sede usada,
- ciudad sede,
- fuente de la sede,
- confianza.

También se agrega diagnóstico en Fuentes:

```text
Sede/estadio: detectada por API-Football / ESPN / calendario / referencia
```

## 6. Clima más coherente

Al cambiar el estadio, el clima y la altitud se calculan con las coordenadas de esa sede, no con una sede genérica fija.

## Cómo actualizar

Reemplaza:

- app_mundial.py
- CAMBIOS_V31.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega deteccion automatica de estadio V31"
git push
```

En Streamlit Cloud:

- Manage app → Reboot app

## Nota

La sede exacta depende de que alguna fuente pública o API entregue el dato. Si no hay dato exacto, la app no inventa: usa referencia profesional y lo informa.
