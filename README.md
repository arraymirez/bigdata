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







