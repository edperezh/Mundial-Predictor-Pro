# Guía rápida de variables avanzadas V7

## Archivo de ratings y alineaciones

Edita:

datos/jugadores_rating.csv

Columnas:
- team: nombre del equipo en inglés como aparece en la app, por ejemplo Colombia, United States, Morocco.
- player: nombre del jugador.
- position: GK, DEF, MID, FWD.
- career_rating: calificación 0 a 10 según carrera.
- current_form_rating: calificación 0 a 10 según forma reciente.
- expected_starter: TRUE si está en alineación oficial/probable/última alineación.
- available: TRUE si está disponible.
- lineup_type: official, probable o last.
- last_seen_date: fecha de la última alineación conocida.
- notes: observaciones.

Si no hay alineación oficial, usa la última alineación reportada o la probable.
El modelo tomará titulares disponibles. Si hay menos de 8, usará los mejores 11 disponibles por rating.

## Variables nuevas en la app

- Temperatura, humedad y altitud.
- Apoyo del público.
- Presión mental contra cada equipo.
- Cohesión interna.
- Polémica/conflicto interno.
- Rating promedio de alineación.
