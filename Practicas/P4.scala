import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("CitiGroup2006_2008")


//1
df.filter($"Close">480).show()
//2
df.select(avg($"Low")).show()
//3
df.filter($"Close" < 480 && $"High" < 480).show()
//4
df.filter($"High"===484.40).show()
//5
df.select(corr($"High", $"Low")).show()
//6
df.select(collect_list($"Open")).collect()
//7
df.first()
//8
df.describe()
//9
df.select(max($"Low")).show()
//10
df.select(approx_count_distinct($"Open")).show()
//11
df.count
//12
df.select(collect_list($"Open")).collect()
//13
df("Close")
//14
df.select(mean(df("Close"))).show()
//15
df.select(stddev(df("Low"))).show()
//16
df.select(min(df("High"))).show()
//17
df.select(month(df("Date"))).show()
//18
df.select(dayofweek(df("Date"))).show()
//19
df.select(weekofyear(df("Date"))).show()
//20
df.printSchema