# Cambios V13 - Mapa profesional y precisión acumulada

## Nuevas mejoras

### 1. Pestaña "Mapa del Mundial"
Agrega un tablero visual profesional con:
- Posiciones por grupo.
- Estados visuales: zona directa, tercero en disputa, riesgo/eliminado.
- Favoritos actuales.
- Ruta de eliminatoria proyectada aproximada.
- Información emergente al pasar el puntero sobre equipos y llaves.
- Diseño visual tipo dashboard.

### 2. Pestaña "Precisión"
Agrega seguimiento acumulado de aciertos:
- Partidos evaluados.
- Porcentaje de acierto en ganador/empate/perdedor.
- Acierto de marcador exacto top 1.
- Acierto de marcador dentro del top 3.
- Tabla partido por partido con predicción vs resultado real.

### 3. Relación con probabilidades existentes
El mapa usa:
- probabilidad base de campeón,
- Monte Carlo si ya se ejecutó,
- posiciones y puntos actuales,
- fuerza estimada del modelo.

## Nota importante
La precisión acumulada es una auditoría operativa aproximada. Sirve para ver cómo va funcionando el modelo con partidos ya jugados, pero no reemplaza una validación estadística pura sin fuga de información.
