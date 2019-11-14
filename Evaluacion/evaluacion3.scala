//importaciones
//a) Importamos de la libreria el Algorimo MultilayerPerceptron
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler

import org.apache.spark.sql.SparkSession


//Creamos la sesion de spark
val spark = SparkSession.builder.appName("Evaluacion3Multilayer").getOrCreate()

//val assembler = new VectorAssembler().setInputCols(Array("hour", "mobile", "userFeatures")).setOutputCol("features")

//Carga la informacion del archivo a DataFrame
var data = spark.read.format("csv").option("header","true").load("iris.csv")

//Para poder procesar los datos,necesitamos una columna llamada label
//que seran las clases para el procesamiento
data = data.withColumnRenamed("species","label") 

//Hacemos condicionales para cambiar las clases existentes en el DF para cambiarlos por valores numericos
//Setosa = 0, Versicolor = 1 y Virginica = 2
data = data.withColumn("label", when(col("label") === "setosa", 0).otherwise(when(col("label") ==="versicolor",1 ).otherwise(2)) )

//Cambiamos las columnas de tipo String a float para poder ser leidas

data = data.withColumn("sepal_length", data("sepal_length").cast("float"))
data = data.withColumn("sepal_width", data("sepal_width").cast("float"))
data = data.withColumn("petal_length", data("petal_length").cast("float"))
data = data.withColumn("petal_width", data("petal_width").cast("float"))


//Convertimos la data del csv a un conjunto de vectores 
//conformado por las columnas de datos a las cuales llamaremos "features"
//para poder ser leida por el Algoritmo.
val assembler = new VectorAssembler().setInputCols(Array("sepal_length", "sepal_width", "petal_length","petal_width")).setOutputCol("features")

//Recibe la data "sucia" y busca las columnas que definimos anteriormente,
//retorna un dataframe con la columna features
var dataClean = assembler.transform(data)

//imprime el esquema corregido a datos numericos
dataClean.printSchema()

//mostramos la data con las columnas corregidas

dataClean.show(10
)
// Separa la informacion en 60% entrenamiento y 40% para pruebas desde la data limpia
val splits = dataClean.randomSplit(Array(0.6, 0.4), seed = 1234L)
val entrenamiento = splits(0)
val test = splits(1)


//b) Disenio de la arquitectura donde especificamos las capas y las neuronas
//4 neuronas en la de entrada, en la capa oculta(intermedia 3) y 3 en la capa de salida
val capas = Array[Int](4, 3, 3)

// Creamos el modelo de entrenamiento para el algoritmo especificando las capas
//el tamanio del bloque para ser calculado, la semilla que servira para calcular los pesos del aloritmo y las iteraciones maximas (100)
val entrenador = new MultilayerPerceptronClassifier().setLayers(capas).setBlockSize(128).setSeed(1234L).setMaxIter(100)



//d)
// Entrena el modelo
//Realiza la el entrenamiento utilizando el calculo de error de backpropagation.
//Calcula la funcion de perdida con respecto a los pesos de la red con base a sus entradas y salidas.
//De esta manera cada vez que se obtienen resultados va buscando el error 'hacia atras'.
//Spark internamente utiliza la "Logistic Loss function" para la optimizacion.
val modelo = entrenador.fit(entrenamiento)

//e)
//Al utilizar backpropagation, se obtiene el error de la ultima capa, 
//entonces pondera el error entre las neuronas que conforman la red.
val result = modelo.transform(test)


val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")


println(s"Exactitud de la prueba = ${evaluator.evaluate(predictionAndLabels)}")