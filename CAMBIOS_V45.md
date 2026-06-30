# Cambios V45 - Nequi claro, modo rápido y tema oscuro

## 1. Correos de ejemplo eliminados

Se eliminaron valores precargados en:

- Correo para recibir link de Mercado Pago.
- Correo para asociar pago Nequi.

Ahora aparecen vacíos con placeholder `tu_correo@ejemplo.com` para evitar que se muestre accidentalmente el correo del administrador.

## 2. Nequi más claro

Se refuerza el mensaje:

- Nequi es manual.
- El acceso no se activa automáticamente.
- La verificación puede tardar más que Mercado Pago.
- Solo se activa después de confirmar el pago real.

## 3. Panel admin para aprobar pagos Nequi

Se agregó la pestaña:

```text
Nequi manual
```

Flujo:

1. El cliente paga por Nequi.
2. Registra correo y referencia.
3. El administrador revisa el movimiento real en Nequi.
4. En Admin > Accesos, clientes y cupones > Nequi manual selecciona el ID pendiente.
5. Presiona `Aprobar Nequi y activar cliente`.
6. El correo queda registrado como cliente pagado.

## 4. Mejora de velocidad

Para evitar cargas de 17-18 minutos:

- La app ya no hace actualización externa pesada en cada rerun.
- Las variables externas de internet quedan desactivadas por defecto.
- El versus usa modo rápido por defecto con datos guardados/locales.
- El admin puede activar actualización externa si realmente la necesita.

## 5. Tema oscuro por defecto

Se agregó:

```text
.streamlit/config.toml
```

con `base="dark"` para que la app abra en modo oscuro por defecto.

## 6. Secrets opcionales nuevos

```toml
AUTO_REFRESH_PUBLIC_RESULTS = "false"
```

Recomendado dejarlo en `false` para velocidad. Si lo pones en `true`, el público puede disparar actualización de resultados al cargar, lo cual puede hacer la app más lenta.
