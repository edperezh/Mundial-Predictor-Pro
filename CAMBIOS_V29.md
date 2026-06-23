# Cambios V29 - Modo producción / inferencia

## Objetivo

La app pública ya no debe reentrenar cada vez ni mostrar controles internos.

## Flujo nuevo

### Usuario público

- Entra a la app.
- La app carga automáticamente `modelos/modelo_oficial.pkl` si existe.
- Puede analizar partidos, ver probabilidades, marcadores, mapa, campeón y precisión.
- No ve botón de reentrenar.
- No ve `Forzar actualización web`.
- No ve campo de API.
- No ve pestañas internas de modelos/matrices/peso de variables.

### Administrador

Se activa desde la barra lateral con:

```toml
APP_ADMIN_CODE = "TU_CODIGO_PRIVADO"
```

El admin puede:

- reentrenar modelo estable,
- forzar actualización web,
- cambiar año base,
- cargar modelo guardado,
- ver comparación de modelos,
- ver matriz de confusión, correlación, peso de variables y datos técnicos.

## Auto-carga del modelo

Al iniciar, la app intenta cargar automáticamente:

```text
modelos/modelo_oficial.pkl
```

Si no existe, el público ve un aviso y el administrador debe entrenar una vez.

## Métricas congeladas

La versión del modelo, fechas de entrenamiento, mejor modelo y métricas quedan asociadas al artefacto guardado.

## Cómo actualizar

Reemplaza:

- app_mundial.py
- requirements.txt
- CAMBIOS_V29.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega modo produccion inferencia V29"
git push
```

En Streamlit Cloud:

- Manage app → Settings → Secrets
- agrega `APP_ADMIN_CODE`
- Reboot app
