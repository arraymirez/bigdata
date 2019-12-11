//pan de cada dia en scala

import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008") //Finantial Crisis


//imprime las primeras 5
df.head(5)

for(row <- df.head(5)){
    println(row)
}


//columns
df.columns

df.describe().show()//describe todos los campos


//Select a column
df.select("Volume").show()

//Select multiple columns
df.select($"Date", $"Close").show()

//create new columns
val df2 = df.withColumn("HighPlusLow", df("High")+df("Low"))

df("High")

df2.printSchema()

df2("HighPlusLow").as("HPL")

df2.select(df2("HighPlusLow").as("HPL"), df("Close")).show()

df.na.drop().show()
df.na.drop(2).show()
df.na.fill(100).show()
df.na.fill("Missing Name")
df.na.fill("New name", Array("Name")).show() 
df.na.fill(200, Array("Sales")).show() 

//df.describe().show()
//df.na.fill(400.5, Array("Sales")).show()
//df.na.fill("Missing name", Array("Name")).show()

//val df2 =df.na.fill(400.5, Array("Sales"))
//df2.na.fill("Missing name", Array("Name")).show()