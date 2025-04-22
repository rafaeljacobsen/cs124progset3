import numpy as np

def repeatedrandom(A,maxiter):
    S=np.random.choice([-1,1],100)
    residue=np.abs(np.dot(A,S))
    for _ in range(maxiter):
        Sp=np.random.choice([-1,1],100)
        residuep=np.abs(np.dot(A,Sp))
        if residuep<residue:
            S=Sp
            residue=residuep
    return np.abs(np.dot(A,S))

def karmarkarkarp(A):
    arr = np.array(A, dtype=np.int64)
    n = len(arr)
    
    while True:
        m1 = np.argmax(arr)
        max_val = arr[m1]
        arr[m1] = -1
        m2 = np.argmax(arr)
        arr[m1] = max_val
        
        if arr[m2] == 0:
            return arr[m1]
            
        diff = arr[m1] - arr[m2]
        arr[m1] = diff
        arr[m2] = 0


def hillclimbing(A,maxiter):
    S=np.random.choice([-1,1],100)
    residue=np.abs(np.dot(A,S))
    for _ in range(maxiter):
        indices=np.random.choice(np.arange(100),2,replace=False)
        i,j=indices[0],indices[1]
        Sp=np.copy(S)
        Sp[i]=-Sp[i]
        if np.random.uniform()<0.5:
            Sp[j]=-Sp[j]
        residuep=np.abs(np.dot(A,Sp))
        if residuep<residue:
            S=Sp
            residue=residuep
    return np.abs(np.dot(A,S))

def T(ite):
    return 10**10 * (0.8)**(ite/300)

def simulatedannealing(A,maxiter):
    S=np.random.choice([-1,1],100)
    residue=np.abs(np.dot(A,S))
    Spp=np.copy(S)
    residuepp=np.abs(np.dot(A,Spp))
    
    probabilities=[]
    for ite in range(maxiter):
        #generate random neighbor Sp
        indices=np.random.choice(np.arange(100),2,replace=False)
        i,j=indices[0],indices[1]
        Sp=np.copy(S)
        Sp[i]=-Sp[i]
        if np.random.uniform()<0.5:
            Sp[j]=-Sp[j]
        residuep=np.abs(np.dot(A,Sp))
        
        #if residue is smaller then S=Sp
        if residuep<residue:
            S=Sp
            residue=residuep
        #otherwise
        else:
            #generate probability
            probability=np.exp(-(residuep-residue)/T(ite))
            probabilities.append(probability)
            #if probability, set S=S'
            if np.random.uniform() < probability:
                S=Sp
                residue=residuep
        if residue < residuepp:
            Spp=S
            residuepp=np.abs(np.dot(A,Spp))
    return(np.abs(np.dot(A,Spp)))

#too slow
def transform(P,A):
    n=len(A)
    Ap=np.zeros(n)
    for j in range(n):
        Ap[P[j]]=Ap[P[j]]+A[j]
    return(int(karmarkarkarp(Ap)))

def repeatedrandomprepartition(A,maxiter):
    n=len(A)
    P=np.random.randint(n,size=n)
    residue=transform(P,A)

    for _ in range(maxiter):
        Pp=np.random.randint(n,size=n)
        residuep=transform(P,A)
        if residuep<residue:
            P=Pp
            residue=residuep
    return residue

def getPneighbor(P,A,n):
    i=np.random.choice(np.arange(n),1)
    while True:
        j=np.random.choice(np.arange(n),1)
        if P[i] != j:
            break
    
    Pp=np.copy(P)
    Pp[i]=j
    residuep=transform(Pp,A)
    return Pp,residuep

def hillclimbingprepartition(A,maxiter):
    n=len(A)
    P=np.random.randint(n,size=n)
    residue=transform(P,A)
    for _ in range(maxiter):
        Pp,residuep=getPneighbor(P,A,n)
        if residuep<residue:
            P=Pp
            residue=residuep
    return residue

def simulatedannealingprepartition(A,maxiter):
    n=len(A)
    P=np.random.randint(n,size=n)
    residue=transform(P,A)
    Ppp=np.copy(P)
    residuepp=transform(Ppp,A)
    
    neighbortime=0
    
    for ite in range(maxiter):
        #generate random neighbor P
        Pp,residuep=getPneighbor(P,A,n)
        
        #if residue is smaller then S=Sp
        if residuep<residue:
            P=Pp
            residue=residuep
        #otherwise
        else:
            #generate probability
            probability=np.exp(-(residuep-residue)/T(ite))
            #if probability, set S=S'
            if np.random.uniform() < probability:
                P=Pp
                residue=residuep
        if residue < residuepp:
            Ppp=P
            residuepp=transform(Ppp,A)
    return(residuepp)
def partition(flag, algorithm, inputfile):
    maxiter=25000
    f = open(inputfile, "r")
    file=f.read()
    A=np.asarray(file.split()).astype(int)
    n=len(A)
    if algorithm == 0:
        return(int(karmarkarkarp(A)))
    if algorithm == 1:
        return(int(repeatedrandom(A,25000)))
    if algorithm == 2:
        return(int(hillclimbing(A,25000)))
    if algorithm == 3:
        return(int(simulatedannealing(A,25000)))
    if algorithm == 11:
        return(int(repeatedrandomprepartition(A,25000)))
    if algorithm == 12:
        return(int(hillclimbingprepartition(A,25000)))
    if algorithm == 13:
        return(int(simulatedannealingprepartition(A,25000)))

import sys
args = sys.argv
print(partition(int(args[1]),int(args[2]),str(args[3])))

