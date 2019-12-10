//Importaciones
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.classification.LinearSVC

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer,VectorAssembler}
import org.apache.spark.sql.SparkSession
    //Creamos la sesion de spark
    val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()
    
    
    //PROCEDIMIENTO DE CARGA DE DATOS

    // Carga los datos a utilizar y los pasa a Dataframe desde el CSV con 40k+ registros
    var dataRaw = spark.read.format("csv").option("inferSchema","True").option("header", "true").option("delimiter", ";").load("Data/bank-additional-full.csv")

    //solo las columnas que analizaremos
    var data = dataRaw.select("age","job","marital","education","default","housing","loan","campaign", "pdays","previous","poutcome","y")
   
    //Se reemplaza la columna por label para poder ser leida por el algoritmo
    //data = data.withColumnRenamed("y","label");
    
    //columnas que no son numericas y que son relevantes para nuestra investigacion
    val colNoNumericas = Array("job","marital","education","default","housing","loan")
        for(c <- colNoNumericas ) yield{ 
           var temp = new StringIndexer().setInputCol(c).setOutputCol(c+"Indexed").fit(data).transform(data);
           data = temp;
        
         }
      
    //Convertimos las columnas restantes a valores numericos 
    val labelpoutcome = new StringIndexer().setInputCol("poutcome").setOutputCol("indexedPoutcome").fit(data)
    val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("label").fit(data)
    //Teniendo los indices, se transforman a valores numericos
    data = labelpoutcome.transform(data);
    data = labelIndexer.transform(data);   
  
    val assembler = new VectorAssembler().setInputCols(Array("age","jobIndexed","maritalIndexed","educationIndexed","defaultIndexed","housingIndexed","loanIndexed", "campaign", "pdays","previous","indexedPoutcome")).setOutputCol("features")
    data = assembler.transform(data);
  
  //FIN DEL PROCEDIMIENTO  DE CARGA Y LIMPIEZA DE DATOS

  //DECISION TREE

    //Indexa las categorias de acuerdo a las combinaciones de features
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(data)       
    // Separamos los datos, 70% para entranemiento y 30% para pruebas.
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3),2048L)
    // Se entrena el arbol de decision
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")
    //convierte las etiquetas indexadas de vuelta a las originales
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    //Indexa las etiquetas a Pipeline
    val pipeline = new Pipeline().setStages(Array(featureIndexer, dt, labelConverter))
    // Entrena el modelo, tambien ajusta los indices.
    val model = pipeline.fit(trainingData)
    // Realiza predicciones.
    val predictions = model.transform(testData)
    // Selecciona y calcula el test error
    val evaluatorDT = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")

//FIN DE DECISION TREE

//MULTILAYER PERCEPTRON

//11 neuronas en la de entrada, 2 capaS ocultas con 5 neuronas y 2 en la capa de salida
val capas = Array[Int](11, 10, 2)
// Creamos el modelo de entrenamiento para el algoritmo especificando las capas
//el tamanio del bloque para ser calculado, la semilla que servira para calcular los pesos del aloritmo y las iteraciones maximas (100)
val entrenador = new MultilayerPerceptronClassifier().setLayers(capas).setBlockSize(128).setSeed(1024L).setMaxIter(100)
//Entrenamos el modelo 
val modeloMLPC = entrenador.fit(trainingData)
//Con el modelo entrenado transformamos los datos de prueba
val resultMLPC = modeloMLPC.transform(testData)
//Seleccionamos las columnas a comparar para la evaluacion de las precicciones.
val predictionAndLabels = resultMLPC.select("prediction", "label")
//Evaluador de MLPC
val evaluatorMLPC = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
//FIN DEL MULTILAYER PERCEPTRON


//LINEAR SUPPORT VECTOR CLASIFIER

val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Se entrena el modelo
val lsvcModel = lsvc.fit(trainingData)
val resultLSVC = lsvcModel.transform(testData)

val predictionLabels = resultLSVC.select("prediction", "label")

val evaluatorLSVC = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")


//FIN DEL LSVC



  //Calculamos la precision de las predicciones
   println(s"Exactiud Decision Tree = ${ evaluatorDT.evaluate(predictions)}")
  println(s"Exactitud MLPC = ${evaluatorMLPC.evaluate(predictionAndLabels)}")
  println(s"Exactitud LSVC = ${evaluatorLSVC.evaluate(predictionLabels)}")
    
    
