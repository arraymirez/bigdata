# K Means


En esta unidad se trabajo con el algoritmo de clustering de K-means (K-medias)

K-Means es uno de los algoritmos mas utilizados de de clustering que agrupa los puntos de datos en un numero 
predefinido de clusters (K) por medio de el calculo de medias



#Clustering

El clustering es una técnica para encontrar y clasificar K grupos de datos (clusters).
Así, los elementos que comparten características semejantes estarán juntos en un mismo grupo,
separados de los otros grupos con los que no comparten características. 

#Funcionamiento del algoritmo
K-means necesita como dato de entrada el número de grupos en los que vamos a segmentar la población.
A partir de este número k de clusters, el algoritmo coloca primero k puntos aleatorios (centroides).
Luego asigna a cualquiera de esos puntos todas las muestras con las distancias más pequeñas.
A continuación, el punto se desplaza a la media de las muestras más cercanas.
Esto generará una nueva asignación de muestras, ya que algunas muestras están ahora más cerca de otro centroide.

Este proceso se repite de forma iterativa y los grupos se van ajustando hasta que la asignación no cambia más moviendo los puntos. 
Este resultado final representa el ajuste que maximiza la distancia entre los distintos grupos y minimiza la distancia intragrupo.

![Ejemplo del funcionamiento de K-Means](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)
