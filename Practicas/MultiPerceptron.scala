

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession


    val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

    ///Carga la informacion del archivo a DataFrame
    val data = spark.read.format("libsvm").load("Data/sample_multiclass_classification_data.txt")

    //Para poder procesar los datos,necesitamos una columna llamada label
    //que seran las clases para el procesamiento
    data.withColumnRenamed("species","label") 




    // Separa la informacion en 60% entrenamiento y 40% para pruebas
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    train.head(5)

    // Especifica las capas para la red neuronal:
    //Capa de entrada de tamanio 4 (caracteristicas), dos intermedias de tamanio 5 y 4
    //Y la salida de tamanio 3 (clases)
    val layers = Array[Int](4, 5, 4, 3)

    // Crea el entrenador y especifica los parametros
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    // Entrena el modelo
    val model = trainer.fit(train)

    // Calcula el error en el conjunto de prueba
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")