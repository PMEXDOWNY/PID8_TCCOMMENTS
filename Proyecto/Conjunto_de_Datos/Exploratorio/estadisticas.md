# Estadísticas del análisis exploratorio

Generado por `exploratorio.py`. No editar a mano.

## 1. Desbalance de clases

**CONDA, intentClass**

```
                  n      %
conda_intent              
O             26603  74.13
E              4711  13.13
A              2299   6.41
I              2274   6.34
```

**Jigsaw, banderas**

```
               n_positivos     %
toxic                15294  9.58
severe_toxic          1595  1.00
obscene               8449  5.29
threat                 478  0.30
insult                7877  4.94
identity_hate         1405  0.88
```

Razón no tóxico : tóxico en CONDA = 4.1 : 1; en Jigsaw = 8.8 : 1.

La clase más rara de Jigsaw es `threat`: 478 ejemplos (0.30%).

## 2. Longitud

```
           count  mean   std  min   50%    90%    99%     max
source                                                       
conda    35887.0   3.3   3.3  1.0   2.0    7.0   16.0    64.0
jigsaw  159571.0  67.3  99.2  1.0  36.0  152.0  567.0  1411.0
```

Caracteres:

```
           count   mean    std  min    50%    90%     99%     max
source                                                           
conda    35887.0   14.9   16.3  1.0   10.0   34.0    75.0   352.0
jigsaw  159571.0  390.9  586.3  6.0  204.0  884.0  3406.3  5000.0
```

Palabras por fuente y etiqueta:

```
                   size  median  mean
source is_toxic                      
conda  0          28902     2.0   3.0
       1           6985     4.0   4.8
jigsaw 0         143346    38.0  68.9
       1          16225    23.0  52.7
```

Mensajes de CONDA de una sola palabra: 35.0%. Comentarios de Jigsaw de una sola palabra: 0.01%.

Jigsaw truncado a 5000 caracteres: 37 comentarios en el tope.

## 3. Co-ocurrencia de banderas en Jigsaw (% condicional, fila → columna)

```
               toxic  severe_toxic  obscene  threat  insult  identity_hate
toxic          100.0          10.4     51.8     2.9    48.0            8.5
severe_toxic   100.0         100.0     95.1     7.0    86.0           19.6
obscene         93.8          18.0    100.0     3.6    72.8           12.2
threat          93.9          23.4     63.0   100.0    64.2           20.5
insult          93.2          17.4     78.1     3.9   100.0           14.7
identity_hate   92.7          22.3     73.5     7.0    82.6          100.0
```

Comentarios con 2 o más banderas: 9865 (6.18% del total; 60.8% de los tóxicos).

## 4. Mensajes más repetidos en CONDA

```
           n  pct_tox
k                    
gg      1834      0.1
lol      933      0.8
ggwp     478      0.2
?        445      0.2
haha     437      0.0
ez       399     99.7
gg wp    370      0.0
xd       194      0.0
ty       183      0.0
hahaha   180      0.6
wp       169      0.6
:d       152      0.0
rofl     113      0.0
wtf      113      7.1
ok        92      0.0
```

Textos distintos en CONDA: 23,951 de 35,887 mensajes. Los 15 más repetidos suman 6092 mensajes (17.0%).

## 5. Vocabulario por clase (log-odds, palabras con ≥30 textos)

Cuenta en cuántos textos aparece cada palabra. Un log-odds alto significa que la palabra aparece mucho más en los tóxicos que en los no tóxicos.

**CONDA (chat de Dota 2): más asociadas a tóxico**

```
          en_toxicos  en_no_toxicos  log_odds
palabra                                      
cunt              64              0      6.28
retarded          56              0      6.15
peru              41              0      5.84
peruvian          36              0      5.71
bitch            103              1      5.65
ez              1543             35      5.19
noobs            100              2      5.11
trash            138              3      5.10
retard           127              3      5.02
idiot            172              5      4.87
pussy             37              1      4.64
noob             678             28      4.59
fuckin            35              1      4.58
faggot            34              1      4.56
suck              98              4      4.51
```

**CONDA (chat de Dota 2): más asociadas a no tóxico**

```
         en_toxicos  en_no_toxicos  log_odds
palabra                                     
glhf              0             45     -3.09
rc                0             41     -3.00
sec               2             96     -2.23
close             1             52     -2.14
wew               4             82     -1.49
ggwp             34            624     -1.48
ty               26            467     -1.45
lel               5             92     -1.40
aw                5             87     -1.35
before            2             37     -1.29
new               2             36     -1.26
hf                4             64     -1.24
lag               6             84     -1.14
fair              2             32     -1.14
comend            2             32     -1.14
```

**Jigsaw (comentarios de Wikipedia): más asociadas a tóxico**

```
               en_toxicos  en_no_toxicos  log_odds
palabra                                           
motherfucker          152              0      7.90
fuckin                135              0      7.78
cocksucker             63              0      7.02
motherfucking          55              0      6.89
cking                  35              0      6.44
fuk                    31              0      6.32
fucker                160              3      6.00
faggot                421              9      5.97
fuck                 2502             76      5.67
fuckin'                46              1      5.61
fucking              1652             55      5.57
bitch                 741             27      5.47
fuckers                36              1      5.37
cock                  302             12      5.37
cunt                  444             19      5.31
```

**Jigsaw (comentarios de Wikipedia): más asociadas a no tóxico**

```
               en_toxicos  en_no_toxicos  log_odds
palabra                                           
subject's               0            479     -4.69
specified               0            474     -4.68
coupled                 0            407     -4.52
specifies               0            309     -4.25
wizard                  0            286     -4.17
selecting               0            281     -4.15
facilitate              0            274     -4.13
conformance             0            272     -4.12
image'                  0            270     -4.11
dropdown                0            241     -4.00
cellpadding             0            238     -3.99
licensing               0            235     -3.98
contributors'           0            227     -3.94
tutorial                2           1088     -3.90
db                      1            640     -3.88
```

## 6. Repetidos y contradicciones

```
         filas  textos_repetidos  etiqueta_conflictiva  % repetidos
source                                                             
conda    35887             13369                  5259         37.3
jigsaw  159571               495                    27          0.3
```

## 7. Distribución por split

```
                size    mean
source split                
conda  train   26914  0.1940
       valid    8973  0.1967
jigsaw train  119679  0.1017
       valid   39892  0.1017
```
