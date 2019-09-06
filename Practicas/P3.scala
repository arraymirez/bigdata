//susecion de fibobacci

//1
def fibo1(num :Int): Int = {
if(num < 2)
     return num
else 
    return (fibo1(num -1 ) + fibo1(num-2))

}

//fibo1(5)




def fibo2(n :Int): Int ={

    if(n<2)
        return n
    else{

        var y = ((1+ Math.sqrt(5))/2)
        var j = ( ( Math.pow(y,n) - Math.pow((1-y),n)) / Math.sqrt(5)    )

        return j.toInt
    }

}

//fibo2(10)

def fibo3(n:Int): Int = {

    var a = 0
    var b = 1
    var c = 0

    for(k <- Array.range(0,n)){

        c = a + b
        a = b
        b = c

    }
    return a
}


def fibo4(n:Int): Int = {

    var a = 0
    var b = 1

    for(k <- Array.range(0,n)){    
        b = b + a
        a = b - a 
    }
    return a
}

def fibo5(n : Int ):Int = {

    if(n<2)
    {
        return n
    }
    else{

        var ar = new Array[Int](n+1)
        ar(0) = 0
        ar(1) = 1

        for(k <- 2 to n ){
            ar(k) = ar(k-1) + ar(k - 2)   
        }

        return ar(n)

    }

}


def fibo6(n : Int): Int = {

    if(n<=0)
        return 0
    else{
        var i = n - 1

        var auxOne = 0
        var auxTwo = 1

        var ab = Array(auxTwo,auxOne)
        var cd = Array(auxOne,auxTwo)


        while(i > 0){
            if(i%2 !=0){
                        //d*b           //c*a
                auxOne = cd(1)*ab(1) +  cd(0)*ab(0)
                auxTwo = ( (cd(1) * (ab(1)+ab(0)))  + cd(0)*ab(1)  )

                ab(0) = auxOne
                ab(1) = auxTwo

            }

            auxOne = ( ( Math.pow(cd(0),2) ).toInt + (Math.pow(cd(1),2) ).toInt )

            auxTwo = ( cd(1)*(2*cd(0) + cd(1) )  )
            
            cd(0) = auxOne 
            cd(1) = auxTwo
            i = i/2
        }
    return ab(0) + ab (1)       

    }
}

