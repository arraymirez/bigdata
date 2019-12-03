//Importamos la session de spark
import org.apache.spark.sql.SparkSession

// Esta linea indica el nivel de rrores que podemos leer, para que aparezcan menos. 
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Se crea una instancia de spark
val spark = SparkSession.builder().getOrCreate()

// Importamos la libreria de Kmeans
import org.apache.spark.ml.clustering.KMeans

// Cargamos la informacion a un dataset
val dataset = spark.read.format("libsvm").load("sample_kmeans_data.txt")


// Entrenamos el modelo de k means con 2 grupos para realizar los clusters
val kmeans = new KMeans().setK(2).setSeed(1L)
val model = kmeans.fit(dataset)

//Evaluamos el clustering con la suma de los errores al cuadrados 
val WSSSE = model.computeCost(dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Mostramos los resultados
println("Cluster Centers: ")
model.clusterCenters.foreach(println)