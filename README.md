
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
=======
![Ejemplo del funcionamiento de K-Means](https://upload.wikimedia.org/wikipedia/commons/e/ea/K-means_convergence.gif)



# Implementacion en spark

```scala
//4.- Importamos la libreria de Kmeans par el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans

//7.- Importamos vector asembler y vector
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

feature_data.columns

//8.- Objeto Vector assembler con las columnas de entrada
val assembler = new VectorAssembler().setInputCols(feature_data.columns).setOutputCol("features")

//9.- Transformamos las columnas del dataset con vector asembler
val data = assembler.transform(dataset_clean)

//10.- Cremamos un modelo k means con k =3
val kmeans = new KMeans().setK(3).setSeed(123L)

//entrenamos el modelo
val model = kmeans.fit(data)


// realizamos las predicciones
val predictions = model.transform(data)

//11.- Evaluamos el modelo con 
val evaluator = new ClusteringEvaluator()

```


# bigdata

# Spark MLib API Introduction

## Machine Learning Algorithms

In this unit we got into the Spark´s machine learning APIs called MLib and ML,
both with similar algorithms but with different implementations. 

Inside the practices folder are the following algorithms, next is a description of the worked
ones:

# Some Libraries

* ## Correlation
Calculating the correlation between two series of data is a common operation in Statistics. 
In spark.ml we provide the flexibility to calculate pairwise correlations among many series.
The supported correlation methods are currently Pearson’s and Spearman’s correlation.

## Iplementation

```scala
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

val df = data.map(Tuple1.apply).toDF("features")
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n $coeff1")

val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n $coeff2")
```

* ## Hypothesis Testing
Hypothesis testing is a powerful tool in statistics to determine whether a result is statistically significant,
whether this result occurred by chance or not. spark.ml currently
supports Pearson’s Chi-squared ( χ2) tests for independence.

ChiSquareTest conducts Pearson’s independence test for every feature against the label.
For each feature, the (feature, label) pairs are converted into a contingency matrix for
which the Chi-squared statistic is computed. All label and feature values must be categorical.

## Iplementation

```scala
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.ChiSquareTest

val df = data.toDF("label", "features")
val chi = ChiSquareTest.test(df, "features", "label").head
println(s"pValues = ${chi.getAs[Vector](0)}")
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
println(s"statistics ${chi.getAs[Vector](2)}")
```

* ## Summarizer
Provides  vector column summary statistics for Dataframe through Summarizer.
Available metrics are the column-wise max, min, mean, variance, and number of nonzeros,
as well as the total count.


## Iplementation

```scala
import org.apache.spark.ml.stat.Summarize


val df = data.toDF("features", "weight")

val (meanVal, varianceVal) = df.select(metrics("mean", "variance")
  .summary($"features", $"weight").as("summary"))
  .select("summary.mean", "summary.variance")
  .as[(Vector, Vector)].first()

println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
  .as[(Vector, Vector)].first()
```


# Machine learning algorithms


* ## Decision tree classifier
Decision trees and their ensembles are popular methods for the machine learning tasks of classification and regression. 
Decision trees are widely used since they are easy to interpret, handle categorical features, extend to the multiclass 
classification setting, do not require feature scaling, and are able to capture non-linearities and feature interactions.
Tree ensemble algorithms such as random forests and boosting are among the top performers for classification and regression tasks.


## Iplementation

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}


// Train a DecisionTree model.
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

```


* ## Random Forest
Random forests are ensembles of decision trees. Random forests combine many decision trees in order to reduce the risk of overfitting. 
The spark.ml implementation supports random forests for binary and multiclass classification and for regression, 
using both continuous and categorical features.

## Iplementation

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}


// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

```


* ## Gradient-boosted tree classifier
Gradient-Boosted Trees (GBTs) are ensembles of decision trees. GBTs iteratively train decision trees in order to minimize a loss function.
The spark.ml implementation supports GBTs for binary classification and for regression,
using both continuous and categorical features.

The following examples load a dataset in LibSVM format, split it into training and test sets, train on the first dataset,
and then evaluate on the held-out test set. We use two feature transformers to prepare the data; these help index categories 
for the label and categorical features, adding metadata to the DataFrame which the tree-based algorithms can recognize.

## Iplementation

```scala
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Train a GBT model.
val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)
  .setFeatureSubsetStrategy("auto")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline.
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)
```

* ## Multilayer Perceptron Classiffer
Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of multiple
layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input data.
All other nodes map inputs to outputs by a linear combination of the inputs with the node’s weights w and bias b and
applying an activation function. This can be written in matrix form for MLPC with K+1 layers as follows:

y(x)=fK(...f2(wT2f1(wT1x+b1)+b2)...+bK)

Nodes in intermediate layers use sigmoid (logistic) function:

    f(zi)=11+e−zi
    
Nodes in the output layer use softmax function:

  f(zi)=ezi∑Nk=1ezk

The number of nodes N in the output layer corresponds to the number of classes.

MLPC employs backpropagation for learning the model.
We use the logistic loss function for optimization and L-BFGS as an optimization routine

## Iplementation

```scala
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

// train the model
val model = trainer.fit(train)

// compute accuracy on the test set
val result = model.transform(test)
```


* ## Linear Support Vector Machine
A support vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space,
which can be used for classification, regression, or other tasks. Intuitively, a good separation is achieved by the
hyperplane that has the largest distance to the nearest training-data points of any class (so-called functional margin),
since in general the larger the margin the lower the generalization error of the classifier.
LinearSVC in Spark ML supports binary classification with linear SVM. Internally, it optimizes the Hinge Loss using OWLQN optimizer.

![Imagen LSVM](https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png)

## Iplementation

```scala
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val lsvc = new LinearSVC()
  .setMaxIter(10)
  .setRegParam(0.1)

// Fit the model
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
```

* ## One vs All
OneVsRest is an example of a machine learning reduction for performing multiclass classification given a base
classifier that can perform binary classification efficiently. It is also known as “One-vs-All.”

OneVsRest is implemented as an Estimator. For the base classifier, it takes instances of Classifier and creates
a binary classification problem for each of the k classes. The classifier for class i is trained to predict whether the label
is i or not, distinguishing class i from all other classes.

Predictions are done by evaluating each binary classifier and the index of the most confident classifier is output as label.

## Iplementation

```scala
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// load data file.
val inputData = spark.read.format("libsvm")
  .load("data/mllib/sample_multiclass_classification_data.txt")

// generate the train/test split.
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// instantiate the base classifier
val classifier = new LogisticRegression()
  .setMaxIter(10)
  .setTol(1E-6)
  .setFitIntercept(true)

// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(classifier)

// train the multiclass model.
val ovrModel = ovr.fit(train)

// score the model on test data.
val predictions = ovrModel.transform(test)

// obtain evaluator.
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

// compute the classification error on test data.
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")
```

* ## Naive Bayes
Naive Bayes classifiers are a family of simple probabilistic, multiclass classifiers based on applying Bayes’
theorem with strong (naive) independence assumptions between every pair of features.

Naive Bayes can be trained very efficiently. With a single pass over the training data,
it computes the conditional probability distribution of each feature given each label. For prediction,
it applies Bayes’ theorem to compute the conditional probability distribution of each label given an observation.

MLlib supports both multinomial naive Bayes and Bernoulli naive Bayes.

Input data: These models are typically used for document classification. Within that context,
each observation is a document and each feature represents a term. A feature’s value is the frequency 
of the term (in multinomial Naive Bayes) or a zero or one indicating whether the term was found in the document 
(in Bernoulli Naive Bayes). Feature values must be non-negative. The model type is selected with an optional
parameter “multinomial” or “bernoulli” with “multinomial” as the default. For document classification, the input 
feature vectors should usually be sparse vectors. Since the training data is only used once, it is not necessary to cache it.

## Iplementation

```scala
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

// Train a NaiveBayes model.
val model = new NaiveBayes()
  .fit(trainingData)

// Select example rows to display.
val predictions = model.transform(testData)
predictions.show()

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")
```




# BIG DATA

# Scala and Spark introduction

## Scala basics

Inside the notes folder we find some sort of code about getting familiar with the
Scala programming language.

As getting the basics with the basics operations, such as:

```scala
//funciones exponenciales
math.pow(4,2)
1+2 * 3+4
(1+2)*(3+4)
```
## Basic operations

```scala
1+1
2-1
2*5
1/2
1/2.0
1.0/2.0
```

## String manipulations

```scala

var lstr ="This is a long string"
lstr.charAt(0)
lstr.charAt(5)
```
## Arrays and other structures

```scala
// Arreglos, son mutables , las lists no
val arr = Array(3,4,5)
val arr = Array("a","b","c")
val arr = Array("a","b", true, 1.2)

//Arreglos con range y saltos
Array.range(0, 10)
Array.range(0, 10, 2)

//Los ocnjuntos no contienen elementos repetidos
val s = Set()
val s = Set(1,2,3)

val s = Set(2,2,2,3,3,3,5,5,5)

val s = collection.mutable.Set(1,2,3)
s += 4

//Mapas
val mymap = Map(("saludo", "Hola"), ("pi", 3.1416), ("z", 1.3))
mymap("pi")
mymap("saludo")
mymap("ja")
mymap get "pi"
mymap get "z"
mymap get "o"

```
# Practices

The practical work is all about getting familiar and putting on practice all the new learned code basics.

Also the installation and configuration of the virtual environment for these new technologies was important 
at the beginning.


# Evaluation

This unit has two evaluations :

1. About a logic problem that has to be solved using our problem resolution skills
  throug scala syntax. 


2. Dataframes
   We worked with dataframes, handle their columns, working with renames, and other functions as follows:
  ```scala
    //3
    println('3')
    df.columns

    //4
    println('4')
    df.printSchema

    //5
    println('5')
    df.head(5)
    //10
    println("10")
    df.select(max($"Volume"),min($"Volume")).show()
    //11
    println("11")
    //a
    println("a")
    df.select($"Close" < 600).count
    println("b")
 ```
  
 

