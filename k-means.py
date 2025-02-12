import numpy as np
arr=np.array([2,4,5,3,15,10,20,30])
k1=arr[0]
k2=arr[1]
op1=np.array([])
op2=np.array([])
iter=0


while True:
    iter=iter + 1 
    op1=np.array([])
    op2=np.array([])
    for i in arr:
        if abs(k1-i)<abs(k2-i) :
            op1 = np.append(op1,i) 


        elif abs(k1-i)==abs(k2-i):
            op1 = np.append(op1,i)

        else :
            op2 = np.append(op2,i) 

    m1=np.mean(op1)
    m2=np.mean(op2)

    if m1==k1:
        break
    k1=m1 
    k2=m2 
print(op1)
print(op2)
print("total iterations =",iter)
