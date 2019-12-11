
//1.Importar una simple sesion de spark 
import org.apache.spark.sql.SparkSession

//2.-importaciones necesarias para las librerias de deteccion de errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

import org.apache.spark.ml.evaluation.ClusteringEvaluator

//imports para funciones sql
import org.apache.spark.sql.functions._


//3.- Creamos una instancia de spark
val spark = SparkSession.builder.appName("KMeansExamen").getOrCreate()

//4.- Importamos la libreria de Kmeans par el algoritmo de agrupamiento
import org.apache.spark.ml.clustering.KMeans


//5.- Cargamos el dataset al 
val dataset = spark.read.format("csv").option("header", "true").load("Wholesalecustomersdata.csv")

//variable para limpiar las colunas
val toInt = udf[Int,String](_.toInt)

dataset.printSchema()

val dataset_clean = dataset.withColumn("Fresh", toInt(dataset("Fresh")) ).withColumn("Milk", toInt(dataset("Milk")) ).withColumn("Grocery", toInt(dataset("Grocery")) ).withColumn("Frozen", toInt(dataset("Frozen")) ).withColumn("Detergents_Paper", toInt(dataset("Detergents_Paper")) ).withColumn("Delicassen", toInt(dataset("Delicassen")) )

dataset_clean.printSchema()

//6.- Seleccionamos las columnas indicadas en un nuevo conjunto
val feature_data=dataset_clean.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")

feature_data.show()

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

val silhouette = evaluator.evaluate(predictions)

val wss = model.computeCost(data);

println(s"Within Set Sum of squared errors  $wss");



//muestra los centros
println("Centroides: ");
model.clusterCenters.foreach(println)

//12. Nombres de las columnas
println("Columnas: " + model.getPredictionCol + " , " + model.getFeaturesCol)
