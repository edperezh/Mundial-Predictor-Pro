# Cambios V39 - Cupones ocultos y reportes cada 4 horas

## 1. Se quitó el ejemplo del cupón

Antes el campo público mostraba:

```text
Ej: BETA-TIKTOK-2026
```

Eso exponía el cupón real. Ahora el campo solo dice:

```text
Ingresa tu cupón
```

## 2. Reporte por cupón en administrador

En:

```text
📊 Analítica → 👥 Accesos, clientes y cupones → Cupones
```

se agregó una tabla de uso por cupón.

La tabla muestra por cada cupón:

- usuarios asociados al cupón,
- clientes pagados con ese cupón,
- pruebas gratis asociadas,
- ingresos a la app de esos usuarios,
- pagos verificados,
- valor verificado,
- veces que el cupón fue escrito/aplicado en pantalla.

## 3. Reporte por correo incluye cupones

El correo de analítica ahora incluye una sección:

```text
REPORTE POR CUPÓN
```

También adjunta un CSV:

```text
reporte_cupones_YYYYMMDD_HHMM.csv
```

## 4. Envío cada 4 horas

El valor recomendado cambió a:

```toml
ANALYTICS_EMAIL_INTERVAL_MINUTES = "240"
```

Recuerda: Streamlit revisa el envío cuando la app recibe una visita o se ejecuta. Si quieres que se envíe estrictamente cada 4 horas aunque nadie entre, usa un ping externo.

## 5. Correo destino

Para enviar los reportes a:

```text
nivisyrafael@gmail.com
```

usa:

```toml
ANALYTICS_EMAIL_TO = "nivisyrafael@gmail.com"
ANALYTICS_EMAIL_INTERVAL_MINUTES = "240"
```

## 6. Precios en COP

Se conservaron:

```toml
MERCADOPAGO_BASE_PRICE = "19900"
MERCADOPAGO_COUPON_PRICE = "15900"
```

## 7. Actualización

Reemplaza:

- app_mundial.py
- CAMBIOS_V39.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega reporte por cupones y envio cada 4 horas V39"
git push
```

Después:

```text
Manage app → Reboot app
```
