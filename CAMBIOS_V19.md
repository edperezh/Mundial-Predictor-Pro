# Cambios V19 - Corrección mensajes rojos de secrets.toml

## Problema

En local aparecían múltiples mensajes rojos:

`No secrets files found. Valid paths for a secrets.toml file are...`

Esto pasaba porque la app intentaba leer `st.secrets` varias veces aunque no existiera `.streamlit/secrets.toml`.

## Solución

Ahora la app primero verifica si existe un archivo:

`.streamlit/secrets.toml`

Solo si existe, lee `st.secrets`.

Si no existe, usa:
- variables de entorno,
- valores por defecto,
- y no muestra errores rojos.

## Importante

Para correr localmente no necesitas `secrets.toml`.

Para subir a Streamlit Cloud y activar paywall, sí puedes configurar Secrets desde el panel de Streamlit Cloud.
