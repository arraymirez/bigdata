

//1 Function
var scores1 = Array(10,5,20,20,4,5,2,25,1)
var games1 = 9


var scores2 = Array(3,4,21,36,10,28,35,5,24,42)
var games2 = 10

//n es el numero de juegos
//scores son las puntuaciones
def breakingRecords(n:Int, scores: Array[Int]): Array[Int] = {

      //validamos que sea la cantidad de puntuaciones correcta
        var minCount = 0
        var maxCount = 0

     var result =Array(minCount,maxCount)
    if(scores.length != n){

        println("El numero de juegos y de puntuaciones son diferentes")
    return result
    }else{

        var min = 0
        var max = 0

        
        for(i <- 0 to scores.length - 1 ){

            if(i == 0){
                min = scores(i)
                max = scores(i)
            }else{

                if(scores(i) < min ){

                    minCount = minCount + 1
                    result(1)= minCount
                    min = scores(i)
                }else if(scores(i) > max ){

                    maxCount = maxCount + 1
                     result(0)=maxCount
                    max = scores(i)
                }


            }


        }

    return result

    }
}

breakingRecords(games2,scores2)

breakingRecords(games1,scores1)


