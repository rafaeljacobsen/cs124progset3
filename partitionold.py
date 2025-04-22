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
    S=np.zeros(100)
    Acopy=np.copy(A)
    while np.sum(Acopy) != np.max(Acopy):
        i,j=np.argpartition(Acopy,2)[-2:]
        Acopy[j]=Acopy[j]-Acopy[i]
        Acopy[i]=0
    return(np.sum(Acopy))


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
    P=np.random.randint(n,size=n)
    Ap=np.zeros(n)
    for j in range(n):
        Ap[P[j]]=Ap[P[j]]+A[j]
    if algorithm == 11:
        return(int(repeatedrandom(Ap,25000)))
    if algorithm == 12:
        return(int(hillclimbing(Ap,25000)))
    if algorithm == 13:
        return(int(simulatedannealing(Ap,25000)))

import sys
args = sys.argv
print(partition(int(args[1]),int(args[2]),str(args[3])))

