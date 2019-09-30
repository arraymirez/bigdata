//1
println('1')
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
//2
println('2')
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv") 

//3
println('3')
df.columns

//4
println('4')
df.printSchema

//5
println('5')
df.head(5)
//6
println('6')
df.describe().show()
//7
println('7')
val df2 = df.withColumn("HV Ratio",($"High")/($"Volume"))

//8
println('8')
println("columna no existe")

//9
println("9")
println("El precio corresponde al precio en que cierran las acciones de la empresa al terminar")
println("la sesion de negociasiones")
//  
//


//10
println("10")
df.select(max($"Volume"),min($"Volume")).show()
//11
println("11")
  //a
  println("a")
    df.select($"Close" < 600).count
println("b")
  //b
    var tiempototal = df.count()
    var tiempomayor = df.filter($"High" > 500.00).count()
    println("Fue mayor el "+ (tiempomayor.toDouble /tiempototal.toDouble) *100.0 + "%")
  //c
  println("c")
  df.select(corr($"High",$"Volume")).show()
  //d
  println("d")
  df.groupBy(year($"Date")).max(df.columns(2)).show()
  //e
  println("e")
  df.groupBy(month($"Date")).mean(df.columns(2)).show()
 
