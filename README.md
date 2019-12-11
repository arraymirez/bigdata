# bigdata



## Proyecto Final

<!-- AUTO-GENERATED-CONTENT:START (TOC:collapse=true&collapseText="Click to expand") -->
<details>
<summary>"Indice..."</summary>

- [Introduccion](#introduccion)
- [Marco teórico](#marco-teorico)
  * [Machine Learning](#machine-learning)
  * [Aprendizaje supervisado y no supervisado](#aprendizaje-supervisado-y-no-supervisado)
  * [Clasificacion](#clasificacion)
  * [Decision Tree](#decision-tree)
  * [Multilayer Perceptron](#multilayer-perceptron)
  * [Support Vector Machine](#support-vector-machine)
- [Implementación](#implementacion)
- [Resultados](#resultados)
- [Conclusiones](#conclusiones)
- [Referencias](#referencias)
  

</details>
<!-- AUTO-GENERATED-CONTENT:END -->

## Introduccion

El presente trabajo tiene como objetivo comparar el rendimiento de diferentes algoritmos de machine learning para la clasificación de grupos de datos de gran tamaño.
Los algoritmos se van a aplicar sobre un conjunto de datos lo más cercano posible a las mediciones obtenidas por una campaña de marketing de manera real.
Una vez aplicado cada uno de los algoritmos se realizará una comparativa y de ello, tratar de obtener información relevante acerca de su rendimiento.

Es importante mencionar que para fines de homogeneidad en las pruebas presentes en este trabajo, todas se realizaron en igualdad de condiciones, es decir, el mismo hardware, los mismos datos para evitar factores externos al momento de realizar las comparativas. 
Además de ello, veremos la forma en que estos algoritmos clasifican los grupos de datos de acuerdo a sus etiquetas.


---

## Marco teorico

Durante el desarrollo de este proyecto estarán presentes algunos conceptos, así como abreviaturas y referencias  a los mismos. Para un mejor entendimiento, a continuación se definirán los conceptos con los cuales se va a trabajar.

### Machine Learning

Es una forma de IA que permite que un sistema aprenda de datos en lugar de a través de programación explícita. 
Sin embargo, no es un proceso simple. Machine Learning  utiliza una variedad de algoritmos que iterativamente “aprenden” de los datos para mejorar, describir datos y predecir resultados.Mientras más datos proceden los algoritmos, es posible producir modelos más precisos basados en esa información. 
La salida de un modelo de machine learning se produce cuando un modelo es entrenado con la información, después del entrenamiento, se pueden realizar un modelo predictivo con datos de la misma categoría.

---

### Aprendizaje supervisado y no supervisado

El aprendizaje no supervisado es un concepto muy profundo que puede abordarse desde diferentes perspectivas, desde psicología y ciencias cognitivas hasta ingeniería. A menudo se llama "aprender sin un maestro".
 Esto significa que el aprendizaje del sistema humano, animal o artificial observa su entorno y,basado en observaciones adapta su comportamiento sin que se le pida que se asocie observaciones a las respuestas deseadas dadas, en oposición al aprendizaje supervisado. 
Un  resultado del aprendizaje no supervisado es una nueva representación o explicación de los datos observados.

---

### Clasificacion

La clasificación es una subcategoría del aprendizaje supervisado en la que el objetivo es predecir las etiquetas de clase categóricas (discreta, valores no ordenados, pertenencia a grupo) de las nuevas instancias, basándonos en observaciones pasadas.[3]

Hay dos tipos principales de clasificaciones:
1. Clasificación Binaria: Es un tipo de clasificación en el que tan solo se pueden asignar dos clases diferentes (0 o 1). El ejemplo típico es la detección de email spam, en la que cada email es: spam → en cuyo caso será etiquetado con un 1 ; o no lo es → etiquetado con un 0.
2. Clasificación Multi-clase: Se pueden asignar múltiples categorías a las observaciones. Como el reconocimiento de caracteres de escritura manual de números (en el que las clases van de 0 a 9).


---

### Decision Tree

Los algoritmos de árbol de decisión desglosan el conjunto de datos mediante la formulación de preguntas hasta conseguir el fragmento de datos adecuado para hacer una predicción.[3]

![DecisionTree](https://s3.amazonaws.com/stackabuse/media/decision-trees-python-scikit-learn-1.png)

Basado en las características de los datos de entrenamiento, el árbol de decisión “aprende” una serie de factores para inferir las etiquetas de clase de los ejemplos.
El nodo de comienzo es la raíz del árbol, y el algoritmo dividirá de forma iterativa el conjunto de datos en la característica que contenga la máxima ganancia de información, hasta que los nodos finales (hojas) sean puros.

---

### Multilayer Perceptron


El Multilayer Perceptron o Perceptrón Multicapa es quizás la arquitectura de red más popular utilizada hoy en dia para clasificacion y regresion. 

Los MLP son redes neuronales alimentadas hacia adelante que generalmente están compuestas por varias capas de nodos con conexiones unidireccionales, a menudo entrenadas por propagación hacia atrás.
El proceso de aprendizaje de la red MLP se basa en las muestras de datos compuestas por el vector de entrada N-dimensional  y el vector de salida deseado, llamado o de salida.

![Neural Network](https://www.researchgate.net/profile/Mohamed_Zahran6/publication/303875065/figure/fig4/AS:371118507610123@1465492955561/A-hypothetical-example-of-Multilayer-Perceptron-Network.pngng)


El Perceptrón multicapa define una relación entre las variables de entrada y las variables de salida de la red. Esta relación se obtiene propagando hacia adelante los valores de las variables de entrada. Para ello, cada neurona de la red procesa la información recibida por sus entradas y produce una respuesta o activación que se propaga, a través de las conexiones correspondientes, hacia las neuronas de la siguiente capa.

---

### Support Vector Machine

Una support vector machine (SVM) aprende la superficie de decisión de dos clases distintas de los puntos de entrada. En muchas aplicaciones cada punto puede no estar asignado completamente a alguna de estas dos clases.[4]
La teoría de las SVM es una nueva técnica de clasificación y ha llamado mucho la atención sobre este tema en los últimos años.

La teoría de SVM se basa en la idea de minimización del riesgo estructural (SRM) . En muchas aplicaciones, se ha demostrado que SVM proporciona un mayor rendimiento que las máquinas de aprendizaje tradicionales  y se ha introducido como herramientas poderosas para resolver problemas de clasificación.

Un SVM primero asigna los puntos de entrada en un espacio de características de  alta dimensiones y encuentra un hiperplano de separación que maximiza El margen entre dos clases en este espacio. 
Maximizando el el margen es un problema de programación cuadrática (QP) y puede ser resuelto a partir de su doble problema mediante la introducción de multiplicadores lagrangianos.

---

## Implementacion

El software utilizado para estas pruebas fue Spark. Apache Spark es un sistema informático distribuido de alto rendimiento y uso general que se ha convertido en el proyecto de código abierto Apache más activo, con más de 1,000 colaboradores activos. [5]

Spark, permite trabajar con Scala, que es una forma abreviada de SCalable LAnguage, se originó en 'École Polytechnique Fédérale de Lausanne' (EPFL), Suiza, en 2003, con el objetivo de lograr un lenguaje de alto rendimiento y altamente concurrente que combine la fuerza de los siguientes dos patrones de programación líderes en la plataforma Java Virtual Machine (JVM): [6]

* Programación orientada a objetos
* programación funcional

APIs muy diferentes, pero algoritmos similares. Estas bibliotecas de aprendizaje automático heredan muchas de las consideraciones de rendimiento de las API RDD y Dataset en las que se basan, 
pero también tienen sus propias consideraciones. MLlib es la primera de las dos bibliotecas y está entrando en un modo de solo mantenimiento / corrección
de errores.

### Conjunto de datos

Los datos están relacionados con campañas de marketing directo de una institución bancaria portuguesa. Las campañas de marketing se basaron en llamadas telefónicas. A menudo, se requería más de un contacto con el mismo cliente para acceder si el producto (deposito bancario a plazo) estaría ('sí') o no ('no') suscrito.

Tomamos todos los ejemplos (41188) y 11 entradas, ordenadas por fecha (de Mayo de 2008 a noviembre de 2010).
De los datos a trabajar, usamos dos categorías de la información proveída y tomamos los siguientes:

Información bancaria del cliente
1. age números
2. job: valores categóricos convertidos a números
3. marital: estado civil categórico convertido a números
4. education: nivel de estudios categóricos convertido a números
5. default: si cuenta con algún tipo de crédito, datos categóricos convertidos a números
6. housing: si renta casa, numerico
7. loan: si tiene algún crédito personal, numérico

Información del cliente relacionada a la campaña actual y el último contacto

8. campaign: cuantas veces se ha contactado al cliente para la campaña actual
9. days: cuántos días han pasado desde el último contacto  de la campaña
10. previous: cuantas veces se ha contactado al cliente anteriormente
11. outcome: cómo han sido los contactos anteriores.



---

## Resultados 
Utilizando los algoritmos descritos anteriormente. Se realizaron las pruebas siguientes de acuerdo al conjunto de datos mencionado anteriormente. Decidí tomar los más de 40 mil datos para tener un resultado lo mayor acertado al resultado final de la empresa de marketing. 
Cada prueba se realizó 5 veces, dado que la distribución de datos se realiza de manera aleatoria. Es decir, en cada corrida de las pruebas los 41188 registros se distribuyeron en un 70% para aprendizaje y 30% para pruebas. De esta manera en cada iteración  se distribuye en datos diferentes.






Iteracion | Decision Tree| Multilayer Perceptron| Support Vector MAchine
------------ | -------------| -------------| -------------
1 | 89.83% | 89.79% | 88.90%
2 | 89.83% | 89.79% | 88.90%
3 | 89.83% | 89.79% | 88.90%
4 | 89.83% | 89.79% | 88.90%
5 | 89.83% | 89.79% | 88.90%
Promedio | 89.83% | 89.79% | 88.90%

Estos resultados arrojaron un nivel muy similar en cada una de las pruebas, si bien al final los decimales si varían (de manera muy pequeña), solo se muestran los primeros 4 para agilizar la lectura de las pruebas.

Durante la realización de estas pruebas, hubo algunos ajustes de parámetros para tratar de mejorar las pruebas.
En SVM se utilizó un parámetro de regresión de 0.1 (10%) y un máximo de 10 iteraciones. Ya que aumentando el parámetro de regresión disminuye el rendimiento, incluso con el valor de 1 bajaba ligeramente el resultado a 88.78%.

Por otra parte el mejor resultado observado para el perceptrón multicapa, se obtuvo debido a la cantidad de capas ocultas utilizadas y de neuronas. 
Para obtener la cantidad de neuronas a analizar tome los siguientes criterios [1]:
* Ser mayor a ⅔ de la cantidad de entradas mas las salidas
* No ser mayor que el doble de cantidad de entradas

EL mejor resultado se obtuvo con 10 neuronas y 1 capa oculta. Ya que dividiendo el resultado con 2 capas ocultas y 4 y 5 neuronas respectivamente, se ve afectado el rendimiento. También, con 9 capas y 1 capa oculta, se comporta de manera deficiente el resultado.
El algoritmo con el mejor resultado fue  decision Tree, ya que sus resultados son más fiables debido a las comparaciones realizadas, de manera condicional, general al final una serie de combinaciones bastante acertadas a los resultados finales, por ello es el algortimo en el que nos conviene 

Otro aspecto a tomar en cuenta en estos algoritmos, es el tiempo de ejecución, ya que Decision Tree tarda menos en ejecutarse que la SVM, aunque este último depende de la cantidad de iteraciones. Una prueba realizada con 100 iteraciones tomó bastante tiempo. Sin embargo, el perceptrón multicapa no es el más rápido, pero tampoco el más lento, pero comparado con SVM es más rápido, ya que en este caso el algoritmo analiza los datos en 100 iteraciones.


---

## Conclusiones

Durante la investigación de datos, el algoritmo más favorable parecía SVM, sin embargo en esta prueba no surgió como el ganador. Esto se debe, según a una investigación, por distintos factores, el primero de ellos es que estamos trabajando solamente con 2 clases, es decir es una comparación binaria, ya que al final se requiere validar solamente 2 clases, entonces el kernel que utiliza spark es una validación lineal al final de cuentas.
Existen otros kernel que se pueden implementar en las SVM pero para un mayor número de clases, lo cual en este caso no era aplicable.
Además de ello, se debe adecuar el conjunto de datos para cada algoritmo y así hacerlo más eficiente, esto puede ser modificando el número de entradas, el número de iteraciones, etc.

Para hacer las pruebas más homogéneas iguale las condiciones de cada algoritmo y así obtener un valor clasificatorio uniforme. Si bien tuvieron un valor similar el rendimiento de cada algoritmo se ve afectado por algunos de los elementos antes mencionados.

Por ello, existe una gran cantidad de algoritmos y muchas aplicaciones, lo fundamental es conocer la operación de cada uno de ellos y saber evaluar sus resultados, incluso poder acoplar los datos de la manera que estos algoritmos se aprovechen al máximo, no cabe duda que el machine learning es una tecnología bastante interesante y con un gran interés por parte de la comunidad científica, el poder analizar correctamente los datos, como en este caso una campaña de marketing, ayuda a la toma de decisiones y sobre todo mejorar las proyecciones de ganancias de cualquier empresa o negocio.

---
## Referencias 


[1] Judith Hurwitz, Daniel Kirsch
Machine Learning for dimmies
John Willey & Sons Inc. , (2018), pp. 1-4
[PDF](https://www.ibm.com/downloads/cas/GB8ZMQZ3)

[2] T.M. Huang, V. Kecman, I. Kopriva
Kernel based algorithms for mining huge data sets, supervised, semi-supervised and unsupervised learning
Springer-Verlag, Berlin, Heidelberg (2006)
[Google Scholar](https://scholar.google.com/scholar_lookup?title=Kernel-based%20methods%20and%20function%20approximation&publication_year=2001&author=G.%20Boudat&author=F.%20Anour)

[3] Roman Victor
Aprendizaje Supervisado: Introducción a la Clasificación y Principales Algoritmos
[Articulo Web](https://medium.com/datos-y-ciencia/aprendizaje-supervisado-introducci%C3%B3n-a-la-clasificaci%C3%B3n-y-principales-algoritmos-dadee99c9407)

[4] Chun-Fu Lin and Sheng-De Wang
IEEE TRANSACTIONS ON NEURAL NETWORKS,
 VOL. 13, NO. 2, MARCH 2002
[PDF](https://www.researchgate.net/profile/Sheng-De_Wang/publication/256309499_Fuzzy_Support_Vector_Machines/links/09e4150bb9692e7d56000000/Fuzzy-Support-Vector-Machines.pdf)

[5] Holden Karau and Rachel Warren
High Performance Spark 
 Copyright © 2017 Holden Karau, Rachel Warren. 

[6] Irfan Elahi
Scala Programming for Big Data Analytics
Notting Hill, VIC, Australia
