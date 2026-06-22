# Cambios V4

- Se corrigió la lógica de sede para Mundial 2026.
- Ya NO existe la opción "Casa del Equipo A" ni "Casa del Equipo B".
- Solo hay ventaja de anfitrión si:
  - United States juega en United States.
  - Mexico juega en Mexico.
  - Canada juega en Canada.
- Los demás equipos no reciben ventaja de casa aunque jueguen en Estados Unidos, México o Canadá.
- Se mantiene la app en español, con botón de ejecución, barra de progreso y autor.

## Ejecutar

```bash
cd /d "C:\Users\USER\Desktop\Betplay\Partidos del mundial"
python -m streamlit run app_mundial.py --server.port 8502
```
