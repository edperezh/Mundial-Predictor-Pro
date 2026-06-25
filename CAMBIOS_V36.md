# Cambios V36 - Registro automático del correo después del pago

## Objetivo

Eliminar el paso manual donde el administrador registraba el correo del cliente después de pagar.

Ahora, si se configura Mercado Pago Checkout Pro, la app puede:

1. crear un enlace de pago para el correo del comprador,
2. redirigir al comprador a Mercado Pago,
3. recibir el retorno del pago,
4. verificar el `payment_id` contra la API de Mercado Pago,
5. registrar automáticamente el correo si el pago está aprobado,
6. enviar automáticamente un código dinámico de ingreso al correo del cliente.

## Importante

Streamlit no es ideal para recibir webhooks permanentes. Por eso esta versión usa el retorno seguro del checkout y verifica el pago con la API antes de registrar el acceso.

Si quieres automatización más robusta después, se recomienda:

- webhook de Mercado Pago,
- backend FastAPI,
- Supabase/Firebase/Postgres,
- endpoint público permanente.

## Nuevo flujo de compra automática

En la pestaña pública:

```text
💳 Comprar acceso
```

si configuras `MERCADOPAGO_ACCESS_TOKEN`, aparecerá:

```text
Correo del comprador
Generar enlace de pago
Abrir checkout seguro
```

Después de pagar, Mercado Pago devuelve al usuario a la app y la app verifica el pago.

## Registro automático

Si el pago está aprobado:

- se registra el correo del comprador,
- se crea o actualiza el cliente como `paid`,
- se guarda el pago en la tabla `payments`,
- se envía un código dinámico al correo,
- el cliente puede ingresar con correo + código.

## Código dinámico

Cada ingreso genera un código nuevo.

- Vence en 10 minutos.
- Se usa una sola vez.
- No se guarda el código real, solo su hash.

## Tabla nueva

Se agregó:

```sql
payments
```

Guarda:

- proveedor
- payment_id
- status
- correo
- monto
- moneda
- cupón
- fecha
- resumen no sensible

## Panel admin

En:

```text
📊 Analítica → 👥 Accesos, clientes y cupones
```

se agregó una pestaña:

```text
Pagos automáticos
```

para ver pagos verificados.

## Secrets nuevos recomendados

```toml
APP_REQUIRE_ACCESS = "true"
APP_BRAND_NAME = "Mundial Predictor Pro"
APP_PUBLIC_URL = "https://predictor-mundial-2026-edp.streamlit.app/"
PRICE_USD = "5.00"
SUPPORT_CONTACT = "tu_correo_o_whatsapp"
APP_ADMIN_CODE = "TU_CODIGO_PRIVADO"

MERCADOPAGO_ACCESS_TOKEN = "APP_USR-..."
MERCADOPAGO_CURRENCY = "COP"
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
MERCADOPAGO_SANDBOX = "false"

ANALYTICS_ENABLED = "true"
ANALYTICS_EMAIL_ENABLED = "true"
ANALYTICS_EMAIL_TO = "nivisyrafael@gmail.com"
ANALYTICS_EMAIL_INTERVAL_MINUTES = "60"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "465"
SMTP_USER = "tu_correo@gmail.com"
SMTP_PASSWORD = "tu_contraseña_de_aplicacion"
SMTP_FROM = "Mundial Predictor Pro <tu_correo@gmail.com>"

APP_ENABLE_SCREEN_PROTECTION = "true"
```

## Nota de moneda

Para Colombia normalmente conviene usar COP en Mercado Pago. Por eso el ejemplo usa:

```toml
MERCADOPAGO_CURRENCY = "COP"
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
```

Si tu cuenta permite USD, puedes usar USD, pero revisa primero en Mercado Pago.

## Cómo actualizar

Reemplaza:

- app_mundial.py
- CAMBIOS_V36.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega registro automatico de clientes por pago V36"
git push
```

Después en Streamlit Cloud:

```text
Manage app → Reboot app
```
