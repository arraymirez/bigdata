# BIG DATA

# Scala and Spark introduction

## Scala basics

Inside the notes folder we find some sort of code about getting familiar with the
Scala programming language.

As getting the basics with the basics operations, such as:

```scala
//funciones exponenciales
math.pow(4,2)
1+2 * 3+4
(1+2)*(3+4)
```
## Basic operations

```scala
1+1
2-1
2*5
1/2
1/2.0
1.0/2.0
```

## String manipulations

```scala

var lstr ="This is a long string"
lstr.charAt(0)
lstr.charAt(5)
```
## Arrays and other structures

```scala
// Arreglos, son mutables , las lists no
val arr = Array(3,4,5)
val arr = Array("a","b","c")
val arr = Array("a","b", true, 1.2)

//Arreglos con range y saltos
Array.range(0, 10)
Array.range(0, 10, 2)

//Los ocnjuntos no contienen elementos repetidos
val s = Set()
val s = Set(1,2,3)

val s = Set(2,2,2,3,3,3,5,5,5)

val s = collection.mutable.Set(1,2,3)
s += 4

//Mapas
val mymap = Map(("saludo", "Hola"), ("pi", 3.1416), ("z", 1.3))
mymap("pi")
mymap("saludo")
mymap("ja")
mymap get "pi"
mymap get "z"
mymap get "o"

```
# Practices

The practical work is all about getting familiar and putting on practice all the new learned code basics.

Also the installation and configuration of the virtual environment for these new technologies was important 
at the beginning.


# Evaluation

This unit has two evaluations :

1. About a logic problem that has to be solved using our problem resolution skills
  throug scala syntax. 


2. Dataframes
   We worked with dataframes, handle their columns, working with renames, and other functions as follows:
  ```scala
    //3
    println('3')
    df.columns

    //4
    println('4')
    df.printSchema

    //5
    println('5')
    df.head(5)
    //10
    println("10")
    df.select(max($"Volume"),min($"Volume")).show()
    //11
    println("11")
    //a
    println("a")
    df.select($"Close" < 600).count
    println("b")
 ```
  
 

