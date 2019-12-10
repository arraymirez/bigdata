import org.apache.spark.sql.SparkSession

val spar = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Sales.csv")

df.printSchema()

//1
df.select(countDistinct("Sales")).show()
//2
df.select(sumDistinct("Sales")).show()
//3
df.select(variance("Sales")).show()
//4
df.select(stddev("Sales")).show()
//5
df.select(collect_set("Sales")).show()
//6
df.groupBy("Company").mean().show()
//7
df.groupBy("Company").count().show()
//8
df.groupBy("Company").max().show()
//9
df.groupBy("Company").min().show()
//10
df.groupBy("Company").sum().show()
//11
df.sort(asc("Sales")).show()
//12
df.sort(desc("Sales")).show()
//13
df.sort(asc("Sales"),desc("Person")).show()
//14
df.sort(asc("Company"),desc("Sales")).show()
//15
df.groupBy("Person").mean().show()
//16
df.groupBy("Person").count().show()
//17
df.groupBy("Person").max().show()
//18
df.groupBy("Person").min().show()
//19
df.groupBy("Person").sum().show()
//20
df.groupBy("Sales").sum().show()