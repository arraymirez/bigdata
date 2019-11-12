//Importaciones
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.SparkSession


    val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

    
    //Carga los datos de entrenamiento de
    val training = spark.read.format("libsvm").load("Data/sample_libsvm_data.txt")


    val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

    // Se entrena el modelo
    val lsvcModel = lsvc.fit(training)

    // Imprime los coeficientes y las interepciones para el  svc
    println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
    
