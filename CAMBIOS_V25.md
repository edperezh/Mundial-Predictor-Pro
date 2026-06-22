# Cambios V25 - Corrección NameError tr

## Error corregido

Al ejecutar aparecía:

`NameError: name 'tr' is not defined`

## Causa

En la V24 quedaron dos llamadas a `tr("retrain", lang)` y `tr("load_saved", lang)` dentro del diccionario `TRANSLATIONS`, antes de que la función `tr()` existiera.

## Solución

Se reemplazaron esas llamadas por textos fijos dentro del diccionario:

- `🚀 Reentrenar modelo estable`
- `📦 Cargar guardado`

## Archivos a subir

- app_mundial.py
- CAMBIOS_V25.md

## Comando

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Corrige traducciones V25"
git push
```

Luego en Streamlit Cloud:

`Manage app → Reboot app`
