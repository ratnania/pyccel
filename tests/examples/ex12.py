from numpy import zeros
from numpy import array
from numpy import dot


a=array([1,2,3,5,8,5],dtype=int)
b=array([5,8,6,9,8,2],dtype=int)
k=zeros(shape=(len(a),len(a)),dtype=int)
d=array([[5,8,6,9,8,2],[5,8,6,9,8,2],[5,8,6,9,8,2],[5,8,6,9,8,2],[5,8,6,9,8,2],[5,8,6,9,8,2]],dtype=int)
print('a = ',a)
print('b = ',b)
print('dot(a,b) = ',dot(a,b))
print('len(a) = ',len(a))
print('len([1,2,3])= ',len([1,2,3]))
print(' k =',k)
x=1.0
t=int()
t=10
#print('factorial(x) = ',factorial(t))
print('sqrt(x) =',sqrt(x))
print('abs(x) = ',abs(x))
print('sin(x) = ',sin(x))
print('cos(pi) = ',cos(pi))
print('exp(x) = ',exp(x))
print('log(x) = ',log(x))
#print('sign(x) = ',sign(x))
#print('csc(x) = ',csc(x))
#print('sec(x) = ',sec(x))
print('tan(x) = ',tan(x))
#print('cot(x) = ',cot(x))
print('asin(x) = ',asin(x))
#print('acsc(x) = ',acsc(x))
print('acos(x) = ',acos(x))
#print('asec(x) = ',asec(x))
print('atan(x) = ',atan(x))
#print('acot(x) = ',acot(x))
print('min(a,b) = ',min(a,b))
print('max(a,b) = ',max(a,b))
#print('min(2,b) = ',min(2,b))
#print('max(1,b) = ',max(1,b))

