# Cambios V44 - Nequi, feedback y precisión eliminatoria

## Mejoras principales

- Agrega opción de pago manual por Nequi.
- El pago por Nequi queda como flujo manual: el acceso solo se activa después de verificar el pago real.
- Agrega pestaña **Sugerencias** para reportar errores, proponer mejoras y calificar la página.
- Los reportes de sugerencias, errores y calificaciones se guardan en base local y se anexan al correo de analítica.
- El correo de analítica ahora incluye una sección de sugerencias y adjunta un CSV de feedback.
- La sección **¿Para qué sirve cada pestaña?** cambia según el usuario:
  - Público: solo muestra las pestañas visibles para clientes.
  - Admin: muestra también secciones técnicas y analítica.
- Corrige la medición de **Precisión** para incluir partidos de fase eliminatoria cuando ya tengan resultado cargado.

## Secrets recomendados

```toml
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
TRIAL_MAX_CONSULTAS = "1"

NEQUI_ENABLED = "true"
NEQUI_PHONE = "TU_NUMERO_NEQUI"
NEQUI_HOLDER = "Mundial Predictor Pro"
NEQUI_INSTRUCTIONS = "Envía el pago por Nequi y registra la referencia para verificación manual."

ANALYTICS_EMAIL_ENABLED = "true"
ANALYTICS_EMAIL_TO = "nivisyrafael@gmail.com"
ANALYTICS_EMAIL_INTERVAL_MINUTES = "240"
```

## Notas

- Nequi no se verifica automáticamente en esta versión.
- Para activar un cliente que pagó por Nequi, verifica el movimiento real y registra el cliente desde el panel administrador.
- Mantener el enfoque como análisis deportivo educativo. No garantiza resultados.
