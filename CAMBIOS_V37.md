# Cambios V37 - Corrección error al validar correo

## Problema corregido

Al solicitar código de prueba o ingreso por correo aparecía:

```text
NameError
```

La causa era que la función `is_valid_email()` usaba `re.match(...)`, pero faltaba importar el módulo `re`.

## Solución

Se agregó:

```python
import re
```

y se reforzó `is_valid_email()` para evitar caídas inesperadas.

## Archivos a reemplazar

- app_mundial.py
- CAMBIOS_V37.md

## Actualización

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Corrige validacion de correo V37"
git push
```

Luego en Streamlit:

```text
Manage app → Reboot app
```
