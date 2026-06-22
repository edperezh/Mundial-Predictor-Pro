# Cambios V6

- Corrige el comportamiento detectado: antes la sede/anfitrión cambiaba los marcadores probables porque afectaba Poisson, pero las probabilidades victoria/empate/derrota se mostraban desde ML puro.
- Ahora las probabilidades de victoria/empate/derrota se calculan desde la matriz final de marcadores, combinando Machine Learning + Poisson + sede/anfitrión.
- La sede ahora afecta goles esperados, marcadores más probables y probabilidad de victoria/empate/derrota.
- Se mantiene la regla correcta: United States solo en United States, Mexico solo en Mexico, Canada solo en Canada.
