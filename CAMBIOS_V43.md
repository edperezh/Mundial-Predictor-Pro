# Cambios V43 - Prueba una vez y ejecución manual de versus

## 1. Prueba gratuita de una sola consulta

La prueba gratuita ahora queda configurada para una sola consulta por correo.

Secrets recomendado:

```toml
TRIAL_MAX_CONSULTAS = "1"
```

Si no se define en Secrets, el código usa 1 por defecto.

## 2. Correos con aviso de Spam

Cuando se envía un código dinámico, la app ahora muestra y envía un recordatorio:

```text
Si no ves el correo en la bandeja principal, revisa Spam, Promociones o Correo no deseado.
```

Esto aplica para:

- código de prueba gratuita,
- código de cliente pagado,
- mensajes posteriores a compra.

## 3. Compra después de agotar la prueba

Cuando el cliente usa su única consulta gratuita e intenta correr otro versus, la app ya no solo muestra un bloqueo. Ahora muestra un formulario para:

- ingresar correo,
- generar link seguro de Mercado Pago,
- abrir checkout.

## 4. El cliente elige manualmente los versus

Antes la app podía cargar un versus por defecto. Ahora:

- Equipo A inicia en `Selecciona un equipo`,
- Equipo B inicia en `Selecciona un equipo`,
- no se corre ningún análisis hasta que el cliente seleccione ambos equipos.

## 5. Ejecución manual con botón

La app no ejecuta el análisis automáticamente. El usuario debe pulsar:

```text
▶️ Ejecutar análisis del versus seleccionado
```

Solo después de ese botón se ejecuta:

- validación de prueba gratuita,
- consulta de contexto,
- predicción del partido,
- resultados y pestañas.

## 6. Seguridad comercial

Se mantiene el lenguaje educativo:

- análisis deportivo,
- probabilidades estadísticas,
- estimación educativa,
- no garantiza resultados,
- no procesa actividades reguladas de juego.

## 7. Actualización recomendada de Secrets

```toml
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
TRIAL_MAX_CONSULTAS = "1"
```

## 8. Prueba local

```powershell
cd "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
python -m streamlit run app_mundial.py --server.port 8502
```

## 9. Subida

```powershell
git add app_mundial.py CAMBIOS_V43.md
git commit -m "V43 prueba una vez y ejecucion manual de versus"
git push
```

Luego en Streamlit Cloud:

```text
Manage app → Reboot app
```
