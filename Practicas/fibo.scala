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


        

    }
}

