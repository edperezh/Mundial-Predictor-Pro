# Guía V16 - Subir a la web y monetizar acceso

## Enfoque seguro

La app ahora tiene una pantalla de acceso tipo paywall. Funciona así:

1. El visitante ve una página pública de venta.
2. Hace clic en "Comprar acceso".
3. Paga en tu link externo de pago.
4. Tú le entregas un código de acceso.
5. El visitante ingresa el código y entra a la plataforma.

## Configuración en Streamlit Cloud

En Streamlit Cloud > Settings > Secrets agrega:

APP_REQUIRE_ACCESS = "true"
APP_BRAND_NAME = "Mundial Predictor Pro"
PRICE_USD = "4.99"
PAYMENT_LINK = "https://tu-link-de-pago"
APP_ACCESS_CODES = "MUNDIAL-2026-001,MUNDIAL-2026-002"
SUPPORT_CONTACT = "soporte@tumarca.com"

## Privacidad

Puedes mostrar una marca o nombre comercial en la página. No necesitas mostrar tu nombre personal dentro de la app.

Pero no existe una forma correcta de hacer que el dinero llegue a una cuenta tuya y que sea imposible saber que es tu cuenta. Las pasarelas de pago, bancos y autoridades pueden requerir verificación de identidad. Usa métodos legales y cumple impuestos.

## Opciones de pago

Para Colombia, puedes usar un link de pago local como Mercado Pago.
Para vender internacionalmente, puedes evaluar un merchant of record o una plataforma que permita cobros en USD.
El link se configura en PAYMENT_LINK.

## Recomendación inicial

Empieza simple:
- precio bajo de entrada,
- acceso por código,
- entrega manual del código,
- luego automatizas con base de datos y webhooks si ves demanda.

## Actualizar después de estar en internet

1. Editas el código local.
2. Subes cambios a GitHub.
3. Streamlit Cloud redeploya la app.
4. Si cambias códigos o link de pago, edita Secrets sin tocar el código.

## Importante

La app debe presentarse como análisis educativo/deportivo. No la presentes como herramienta para apostar ni prometas resultados garantizados.
