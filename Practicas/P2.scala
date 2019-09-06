//Practice 2
// 1. Crea una lista llamad "lista" con los elementos "rojo", "blanco", "negro"
var lista = List("rojo", "blanco","negro")


// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"

var clista = collection.mutable.Set(lista:_*)

clista += ("lista","verde","amarillo","azul","naranja","perla")

// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"

val busqueda = Set("lista","verde","amarillo","azul")
clista.&(busqueda)
// 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5

var ar = Array.range(1,1000,5)

// 5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos
val lis = List(1,3,3,4,6,7,3,7)

val lset = lis.toSet
lset


// 6. Crea una mapa mutable llamado nombres que contenga los siguiente
//     "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"

val mutmapa = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23), ("Susana", 27))
// 6 a . Imprime todas la llaves del mapa

  mutmapa.keys
// 7 b . Agrega el siguiente valor al mapa("Miguel", 23)

mutmapa += ("Miguel" -> 23)