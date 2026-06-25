# Cambios V38 - Precios en COP y eliminación de autor visible

## 1. Precios corregidos en la interfaz

La app ya no muestra el precio base como USD 5 cuando Mercado Pago está configurado en COP.

Ahora muestra:

```text
Precio base: COP $19.900
Precio con cupón: COP $15.900
Precio en checkout: COP $19.900 o COP $15.900
```

## 2. Defaults de Mercado Pago

Se ajustaron los valores por defecto:

```toml
MERCADOPAGO_CURRENCY = "COP"
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
```

## 3. Formato de moneda

Se agregó `format_money()` para mostrar:

- COP sin decimales y con separador de miles.
- USD con dos decimales si alguna vez se usa.

## 4. Cupón

El cupón ahora usa el precio real del checkout:

- sin cupón: `MERCADOPAGO_BASE_PRICE`
- con cupón: `MERCADOPAGO_COUPON_PRICE`

## 5. Autor removido

Se quitó el nombre del autor visible de:

- encabezado principal,
- barra lateral,
- comentarios superiores del archivo.

La app queda marcada como:

```text
Mundial Predictor Pro
```

## 6. Secrets recomendados

```toml
APP_REQUIRE_ACCESS = "true"
APP_BRAND_NAME = "Mundial Predictor Pro"
APP_PUBLIC_URL = "https://predictor-mundial-2026-edp.streamlit.app/"
PRICE_USD = "5.00"
SUPPORT_CONTACT = "nivisyrafael@gmail.com"
APP_ADMIN_CODE = "TU_CODIGO_PRIVADO"

MERCADOPAGO_ACCESS_TOKEN = "APP_USR-..."
MERCADOPAGO_CURRENCY = "COP"
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
MERCADOPAGO_SANDBOX = "false"

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = "465"
SMTP_USER = "nivisyrafael@gmail.com"
SMTP_PASSWORD = "TU_CONTRASEÑA_DE_APLICACION_GMAIL"
SMTP_FROM = "Mundial Predictor Pro <nivisyrafael@gmail.com>"
```

## 7. Actualización

Reemplaza:

- app_mundial.py
- CAMBIOS_V38.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Corrige precios COP y quita autor V38"
git push
```

Después:

```text
Manage app → Reboot app
```
