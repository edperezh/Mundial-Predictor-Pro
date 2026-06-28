# Cambios V40 - Prueba gratuita de 5 consultas y nuevos precios

## 1. Prueba gratuita limitada a 5 consultas

Ahora cada correo puede usar la prueba gratuita para máximo:

```toml
TRIAL_MAX_CONSULTAS = "5"
```

Una consulta se cuenta cuando el usuario analiza un partido/equipos.

Reglas:

- Cada correo tiene 5 consultas gratuitas.
- Repetir el mismo partido no descuenta otra consulta.
- Invertir los equipos A vs B / B vs A cuenta como el mismo partido.
- Cuando llega a 5 consultas, la app bloquea nuevos análisis y muestra mensaje para comprar el acceso completo.
- El código dinámico de correo se mantiene, pero ya no da acceso ilimitado.

## 2. Nuevas tablas/columnas

Se agregó:

```sql
trial_queries
```

Se agregan automáticamente columnas a `customers` si faltan:

```sql
trial_queries_used
trial_max_queries
```

## 3. Nuevos precios

Precio sin cupón:

```toml
MERCADOPAGO_BASE_PRICE = "14900"
```

Precio con cupón:

```toml
MERCADOPAGO_COUPON_PRICE = "10900"
```

## 4. Secrets recomendados

```toml
APP_REQUIRE_ACCESS = "true"
APP_BRAND_NAME = "Mundial Predictor Pro"
APP_PUBLIC_URL = "https://predictor-mundial-2026-edp.streamlit.app/"
PRICE_USD = "5.00"
SUPPORT_CONTACT = "nivisyrafael@gmail.com"
APP_ADMIN_CODE = "TU_CODIGO_PRIVADO"

TRIAL_MAX_CONSULTAS = "5"

MERCADOPAGO_ACCESS_TOKEN = "APP_USR-..."
MERCADOPAGO_CURRENCY = "COP"
MERCADOPAGO_BASE_PRICE = "14900"
MERCADOPAGO_COUPON_PRICE = "10900"
MERCADOPAGO_SANDBOX = "false"

ANALYTICS_ENABLED = "true"
ANALYTICS_EMAIL_ENABLED = "true"
ANALYTICS_EMAIL_TO = "nivisyrafael@gmail.com"
ANALYTICS_EMAIL_INTERVAL_MINUTES = "240"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "465"
SMTP_USER = "nivisyrafael@gmail.com"
SMTP_PASSWORD = "TU_CONTRASEÑA_DE_APLICACION_GMAIL"
SMTP_FROM = "Mundial Predictor Pro <nivisyrafael@gmail.com>"

APP_ENABLE_SCREEN_PROTECTION = "true"
```

## 5. Actualización

Reemplaza:

- `app_mundial.py`
- `CAMBIOS_V40.md`

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Limita prueba gratis a 5 consultas y ajusta precios V40"
git push
```

Después:

```text
Manage app → Reboot app
```
