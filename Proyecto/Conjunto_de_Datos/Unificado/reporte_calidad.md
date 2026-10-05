# Reporte de calidad

```
CONDA: 35895 filas leidas (train+valid; test excluido por no tener etiquetas)
CONDA: -8 sin texto
CONDA: -0 vacios tras limpiar
Jigsaw: 159571 filas leidas
Jigsaw: -0 vacios tras limpiar
Jigsaw: 931 con alguna etiqueta tóxica pero toxic=0 (se marcan is_toxic=1)

UNIFICADO: 195458 filas

Filas por fuente y split:
split    train  valid
source               
conda    26914   8973
jigsaw  119679  39892

% toxico por fuente y split:
split   train  valid
source              
conda   19.40  19.67
jigsaw  10.17  10.17

Longitud (caracteres) por fuente:
           count   mean    std  min   25%    50%    75%     max
source                                                         
conda    35887.0   14.9   16.3  1.0   4.0   10.0   20.0   352.0
jigsaw  159571.0  390.9  586.3  6.0  94.0  204.0  432.0  5000.0

Longitud (palabras) por fuente:
           count  mean   std  min   25%   50%   75%     max
source                                                     
conda    35887.0   3.3   3.3  1.0   1.0   2.0   4.0    64.0
jigsaw  159571.0  67.3  99.2  1.0  17.0  36.0  75.0  1411.0

Textos repetidos (is_dup_text=1): 13864 filas
Textos repetidos con etiqueta contradictoria (is_conflict_text=1): 5286 filas
Textos presentes en ambas fuentes: 2
CONDA: textos que aparecen en train y valid a la vez: 799
```
