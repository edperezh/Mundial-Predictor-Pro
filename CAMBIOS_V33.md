# Cambios V33 - Privacidad y analítica legal básica

## Objetivo

Agregar una política de privacidad visible y coherente con la analítica anónima de la app.

## Qué se agregó

### 1. Nueva sección pública de privacidad

Ahora la app incluye una pestaña:

```text
🔒 Privacidad
```

Visible para usuarios públicos y administradores.

### 2. Política bilingüe

La política aparece en inglés y español, útil para vender globalmente.

Incluye:

- qué eventos anónimos se recopilan,
- qué datos NO se guardan directamente,
- cómo se manejan pagos,
- cómo se manejan códigos de acceso,
- finalidad de la analítica,
- aviso deportivo/educativo.

### 3. Aviso en el paywall

En la pantalla de acceso/pago, el expander de privacidad ahora muestra la política completa y aclara que:

- la app no guarda IP, nombres, correos, tarjetas, contraseñas, API keys ni códigos ingresados;
- los procesadores de pago y hosting pueden tener sus propias políticas.

### 4. Evento anónimo de vista de privacidad

Se agregó evento:

```text
privacy_policy_view
```

Este evento ayuda a saber si los usuarios abren la política de privacidad. No guarda datos personales.

### 5. Analítica más clara

El panel de analítica ahora explica mejor que solo guarda eventos anónimos por sesión.

## Eventos anónimos recomendados

La app puede registrar:

- session_start
- paywall_view
- payment_intent_click
- access_granted
- access_denied
- admin_login
- privacy_policy_view

## Datos que no guarda directamente

La app informa que no guarda directamente en analítica interna:

- nombres
- correos
- tarjetas
- contraseñas
- API keys
- códigos ingresados
- ubicación exacta
- IP

## Cómo actualizar

Reemplaza:

- app_mundial.py
- CAMBIOS_V33.md

Luego:

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add .
git commit -m "Agrega politica de privacidad y analitica legal V33"
git push
```

En Streamlit Cloud:

```text
Manage app → Reboot app
```

## Secret recomendado

Puedes dejar:

```toml
ANALYTICS_ENABLED = "true"
```

