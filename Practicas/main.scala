.//Practica 1
//1. Desarrollar un algoritmo en scala que calcule el radio de un circulo

var circunferencia = 15

println("1.- El radio es: " + circunferencia/(2*math.Pi))


//2. Desarrollar un algoritmo en scala que me diga si un numero es primo
var numero=7


println("2.- El numero "+ (if(7%2 == 0)"es" else "no es ") + "primo" )

//3. Dada la variable bird = "tweet", utiliza interpolacion de string para
//   imprimir "Estoy ecribiendo un tweet"
var bird="tweet"

println("3.- Estoy escribiendo un "+bird)

//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slilce para extraer la
//   secuencia "Luke"
var mensaje = "Hola Luke yo soy tu padre!"
println("4.- "+mensaje.slice(5,9))

//5. Cual es la diferencia en value y una variable en scala?
//R: Value es un valor inmutable o constante, ya no puede tener otro valor en tiempo de ejecucion
//a diferencia de una variable que si puede tener distintos valores.

//6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23))) regresa el numero 3.1416 

var tupla =((2,4,5),(1,2,3),(3.1416,23))
tupla._3._1
