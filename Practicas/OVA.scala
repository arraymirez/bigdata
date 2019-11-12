//Importaciones
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
    //Creamos sesion de spark
    val spark = SparkSession.builder.appName(s"OneVsRestExample").getOrCreate()

    
    //Carga El archivo de datos de SVM a Dataframe
    val inputData = spark.read.format("libsvm").load("Data/sample_multiclass_classification_data.txt")

    // Separamos los datos en 80% entrenamiento y 20% pruebas
    val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

    // Se debe instanciar el clasificador base, ya que OvsA utiliza otro clasificador
    val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

    // Instanciamos el clasificador ova
    val ovr = new OneVsRest().setClassifier(classifier)

    // Se entrena el modelo multiclase
    val ovrModel = ovr.fit(train)

    // Evaluamos las predicciones con los datos de prueba
    val predictions = ovrModel.transform(test)

    // Evaluamos la precision del predicotr
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    //Calculamos el error
    val accuracy = evaluator.evaluate(predictions)
    
    println(s"Test Error = ${1 - accuracy}")