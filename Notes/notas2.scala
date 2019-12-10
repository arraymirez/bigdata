// Arreglos, son mutables , las lists no
val arr = Array(3,4,5)
val arr = Array("a","b","c")
val arr = Array("a","b", true, 1.2)

//Arreglos con range y saltos
Array.range(0, 10)
Array.range(0, 10, 2)

Range(0,5)

//Los ocnjuntos no contienen elementos repetidos
val s = Set()
val s = Set(1,2,3)

val s = Set(2,2,2,3,3,3,5,5,5)

val s = collection.mutable.Set(1,2,3)
s += 4

val ims = collection.mutable.Set(2,3,4)
ims += 5
ims.add(6)
ims

ims.max
ims.min

val mylist = List(1,2,3,1,2,3)
mylist.toSet

val newset = mylist.toSet
newset

//Mapas

val mymap = Map(("saludo", "Hola"), ("pi", 3.1416), ("z", 1.3))
mymap("pi")
mymap("saludo")
mymap("ja")
mymap get "pi"
mymap get "z"
mymap get "o"

val mutmap = collection.mutable.Map(("z", 123), ("a", 5), ("b", 7))

mutmap += ("lucky" -> 777)
mutmap
mutmap.keys
mutmap.values