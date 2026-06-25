# Cambios V35 - Compliance, prueba gratuita, cupones y accesos por correo

## 1. Limpieza de lenguaje sensible

Se corrigieron textos visibles para reducir palabras que podrían aumentar rechazo en pasarelas de pago.

Se evitó lenguaje como:

- apuestas
- apostar
- betting
- gambling
- odds
- tipster
- ganancias
- resultado seguro
- pronóstico garantizado

Se reemplazó por lenguaje más seguro:

- análisis deportivo educativo
- probabilidades estadísticas
- simulaciones
- datos del Mundial 2026
- no garantiza resultados
- no procesa pagos de juego ni actividades reguladas

## 2. Prueba gratuita una sola vez

La app ahora permite una prueba gratuita por correo.

Flujo:

1. Usuario escribe su correo.
2. La app verifica si ya usó la prueba.
3. Si no la ha usado, genera un código dinámico.
4. El código se envía al correo.
5. El usuario ingresa el código.
6. Obtiene acceso de prueba.

Nota: controlar “una vez por dispositivo” no es totalmente confiable en una app web, porque el usuario puede cambiar navegador, borrar datos o usar otro equipo. Por eso se implementó control por correo, que es más estable y administrable.

## 3. Código dinámico por inicio de sesión

Para clientes pagados:

1. El administrador registra el correo del cliente.
2. El cliente solicita ingreso con su correo.
3. La app genera un código nuevo.
4. La app envía el código al correo del cliente.
5. El código vence en 10 minutos y cambia cada vez.

El código real no se guarda en la base de datos; solo se guarda su hash.

## 4. Cupones de descuento

Se agregó sistema de cupones.

Cupón inicial:

```text
BETA-TIKTOK-2026
```

Precio promocional:

```text
USD 4.00
```

Precio base recomendado:

```text
USD 5.00
```

Desde el panel admin puedes crear/actualizar cupones.

## 5. Registro administrativo

En la pestaña:

```text
📊 Analítica
```

se agregó:

```text
👥 Accesos, clientes y cupones
```

Incluye:

- registrar cliente pagado
- ver clientes
- crear cupones
- ver cupones
- ver eventos de acceso
- descargar CSV de clientes

## 6. Protección visual contra copia

Se agregó una protección ligera:

- marca de agua con correo enmascarado
- bloqueo de clic derecho
- bloqueo básico de Ctrl+S, Ctrl+P, Ctrl+U
- blur al intentar imprimir

Importante: ninguna página web puede impedir totalmente capturas de pantalla del sistema operativo o de otro celular. La medida es disuasiva, no absoluta.

## 7. Secrets recomendados

```toml
APP_REQUIRE_ACCESS = "true"
APP_BRAND_NAME = "Mundial Predictor Pro"
PRICE_USD = "5.00"
PAYMENT_LINK = "AQUI_TU_LINK_DE_PAGO"
APP_PUBLIC_URL = "https://predictor-mundial-2026-edp.streamlit.app/"
SUPPORT_CONTACT = "tu_correo_o_whatsapp"
APP_ADMIN_CODE = "TU_CODIGO_PRIVADO"

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

## 8. SMTP

Para Gmail debes usar contraseña de aplicación, no tu contraseña normal.

## 9. Actualización

Reemplaza:

- app_mundial.py
- CAMBIOS_V35.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega accesos por correo cupones y compliance V35"
git push
```

Después en Streamlit Cloud:

```text
Manage app → Reboot app
```
