# Cambios V42 - Mapa FIFA cargado y eliminados en 0%

## Cambios principales

- Los equipos eliminados quedan con `prob_campeon_% = 0`.
- Los equipos vivos se renormalizan para que el total sea 100%.
- Se elimina el cuadro duplicado `Dieciseisavos proyectados estilo cuadro oficial`.
- El mapa principal ahora muestra solo un `Mapa completo proyectado de eliminatorias`.
- El mapa usa una formación de llaves basada en el cuadro oficial cargado de la fase eliminatoria.
- Cada versus pendiente se calcula automáticamente con el modelo 1X2.
- Si un partido ya tiene resultado cargado, avanza el ganador real y el perdedor queda eliminado.
- Se conserva la aclaración: es una estimación estadística, no garantiza resultados.

## Archivos a reemplazar

- `app_mundial.py`
- `CAMBIOS_V42.md`

## Comandos

```powershell
cd "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
git add app_mundial.py CAMBIOS_V42.md
git commit -m "V42 mapa oficial y eliminados en cero"
git push
```
