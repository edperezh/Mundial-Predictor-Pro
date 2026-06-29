# V41 - Mejoras pre lanzamiento

## Cambios principales

1. Autoactualización de resultados del Mundial 2026:
   - Revisa calendario interno, CSV manual, fuentes públicas gratuitas y API-Football si está configurada.
   - Actualiza resultados cargados sin reentrenar públicamente.
   - Recalcula memoria estadística para tablas, precisión y mapa.

2. Precios actualizados:
   - Sin cupón: COP $19.900.
   - Con cupón: COP $15.900.

3. Prueba gratis:
   - Pasa de 5 a 2 consultas por correo.

4. Analítica:
   - "Días registrados" ahora se muestra como "Días acumulados" desde la primera visita.
   - Mantiene días activos en la ayuda de la métrica.

5. Mapa del Mundial:
   - Se mantiene tabla de grupos y favoritos.
   - Se cambia el texto de eliminatoria aproximada por dieciseisavos proyectados estilo cuadro oficial.
   - Se agrega mapa completo proyectado: dieciseisavos, octavos, cuartos, semifinal y final.
   - El ganador proyectado se muestra como estimación estadística, no como garantía.

6. Explicación de pestañas:
   - Se agrega un expander “¿Para qué sirve cada pestaña?” antes de las pestañas principales.

## Secrets necesarios

```toml
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
TRIAL_MAX_CONSULTAS = "2"
```

## Actualización segura

Esta versión solo reemplaza el repositorio de la app Streamlit `Mundial-Predictor-Pro`.
No toca el repositorio de la landing `mundial-predictor-landing` ni el dominio de Vercel.
