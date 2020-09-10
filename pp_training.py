# -*-coding:utf-8 -*-
import pandas as pd 
import numpy as np
from all_ss_module import *
import argparse
import math
import struct
import sys
import time
import warnings
import math 
from compiler.ast import flatten

def process_a_b(a, b):

    if a > 0:
        b += np.floor(a)
        a -= np.floor(a)
    elif a<0:
        a += np.floor(b)
        b -= np.floor(b)

    if a>=1:
        b=0
        a=1-1e-10
    if b>=1:
        a=0
        b=1-1e-10
    
    if(abs(a) >1 or abs(b)>1):
        print a,b

    assert(abs(a)<=1)
    assert(abs(b)<=1)

    return a, b

def RBF_linear(x1_a, x1_b, x2_a, x2_b , N, gamma = 0.5):
    bias = 0.02

    ans1, ans2 = InnerProductss(x1_a, x1_b, x2_a, x2_b)
    # res1, res2 = NumProductss(ans1+bias, ans2+bias, ans1+bias, ans2+bias)
    ans1, ans2 = process_a_b(ans1, ans2)

    return ans1, ans2


def RBF_Gaussian(x1_a, x1_b, x2_a, x2_b , N, gamma = 0.5):

    s_a = 0
    s_b = 0
    x3_a = x1_a - x2_a
    x3_b = x1_b - x2_b 

    s_a, s_b = InnerProductss(x3_a, x3_b, x3_a, x3_b)

    assert ( ( (s_a + s_b) - (x3_a+x3_b).dot((x3_a+x3_b))) < 1e-1   )

    f_a = -0.5 * gamma * s_a
    f_b = -0.5 * gamma * s_b


    f_a = np.exp( f_a )
    f_b = np.exp( f_b )
    

    f_a_a = random.random()
    f_a_b = f_a - f_a_a

    f_b_a = random.random()
    f_b_b = f_b - f_b_a


    e_a, e_b = NumProductss(f_a_a, f_a_b, f_b_a, f_b_b)

    e_a, e_b = process_a_b(e_a, e_b)

    return e_a, e_b

class SVM:
    def __init__(self, C, lr=0.1):

        self.C = C
        self.lr = lr
        self.alpha_a = None
        self.alpha_b = None
        self.b_a = 0
        self.b_b = 0
        self.X_a = None
        self.X_b = None
        self.loss_a = float("inf")
        self.loss_b = float("inf")
       
    def predict(self, x, raw=False):

    	# x_a = np.random.randint(1,10, (1,9))
        x_a = np.zeros(len(x))
        for k in range(len(x)):
            x_a[k] = np.random.uniform(0, x[k])

        x_b = x - x_a
        # y_pred = 0+self.b
        y_pred_a = 0+self.b_a
        y_pred_b = 0+self.b_b
        N = len(self.X)

        for i in range(N):

            rbf_a, rbf_b = RBF_Gaussian(x_a, x_b, self.X_a[i], self.X_b[i], N )

            y_add_a, y_add_b = NumProductss(rbf_a, rbf_b, self.alpha_a[i], self.alpha_b[i])
            y_pred_a += y_add_a
            y_pred_b += y_add_b
            y_pred = y_pred_a + y_pred_b

        if raw:
            return y_pred
        return np.sign(y_pred).astype(np.float32)
        
        
    def fit(self, X, y, iteration_times = 500, batch_size=10):

        lr = self.lr
        N = len(X)
        # X_a = np.random.randint(1,10, (N,9))
        X_a = np.zeros((N,9))

        for i in range(N):
            for j in range(9):
                X_a[i][j] = np.random.uniform(0, X[i][j])      
        X_b = X - X_a

        self.X = X
        self.X_a = X_a
        self.X_b = X_b

        y_a = np.zeros(len(y))
        
        for ii in range(len(y)):
            r = np.random.uniform(0,1)
            if y[ii] == 1:
                y_a[ii] = r
            else:
                y_a[ii] = 0 - r
        y_b = y - y_a 
        
        self.alpha_a = np.random.uniform(0,0.5, N)    #*self.C*0.01
        self.alpha_b = np.random.uniform(0,0.5, N)    #*self.C*0.01

        #计算核矩阵
        K_a = np.zeros((N,N))
        K_b = np.zeros((N,N))

        for i in range(N):
            for j in range(N):
                K_a[i][j], K_b[i][j] = RBF_Gaussian(X_a[i],X_b[i],X_a[j],X_b[j],N)  
        

        
        print 'Kernel Matrix Done', 

        K_diag_a = np.diag(K_a)
        K_diag_b = np.diag(K_b)


        for t in range(iteration_times):
            a_a = self.alpha_a.reshape(1,-1)
            a_b = self.alpha_b.reshape(1,-1)

            # loss = a.dot(K).dot(a.T)
            F_a, F_b = MatMulss(a_a, a_b, K_a, K_b)
            a_a_T = np.transpose(a_a)
            a_b_T = np.transpose(a_b)
            loss_a = 0 
            loss_b = 0

            da_a = np.zeros(N)
            da_b = np.zeros(N)
            db_a = 0
            db_b = 0

            P_a = K_a + np.transpose(K_a)
            P_b = K_b + np.transpose(K_b)

            H_a = np.zeros(N)
            H_b = np.zeros(N)

            for i in range(0,N):
             	self.alpha_a[i] -= H_a[i]
             	self.alpha_b[i] -= H_b[i]

            indices = np.random.permutation(N)[:batch_size]
            #print  indices
            for i in indices:

                q_a, q_b = InnerProductss(self.alpha_a, self.alpha_b, K_a[i], K_b[i])

                q_a = q_a + self.b_a
                q_b = q_b + self.b_b

                w_a, w_b = NumProductss(y_a[i], y_b[i], q_a, q_b)

                w_a = 0.5 - w_a
                w_b = 0.5 - w_b

                w_a_int = int(math.floor(w_a * 10000))
                w_b_int = int(math.floor(w_b * 10000))

                margin = w_a + w_b

                u_1, u_2 = BitExtractionMatrix2(w_a_int, w_b_int, 1, 1)       # comparison
                u_sum = u_1^u_2

                loss_a += self.C*w_a
                loss_b += self.C*w_b

                if (u_sum == 0) and (margin < 0):
                    print 'w_a',w_a,'w_b', w_b
                    print 'w_a_int',w_a_int,'w_b_int',w_b_int
                    print 'margin',margin,'u_sum',u_sum

                if u_sum == 0:
                    yk_a, yk_b = DotProductss(y_a[i], y_b[i], K_a[i], K_b[i])
                    da_a -= self.C*yk_a
                    da_b -= self.C*yk_b
                    db_a -= self.C*y_a[i]
                    db_b -= self.C*y_b[i]
                


            if (t+1)%500==0:
                print("Iteration %d, " %(t+1),"Loss:",loss_a+loss_b)
            self.alpha_a -= lr*da_a
            self.alpha_b -= lr*da_b
            self.b_a -= lr*db_a
            self.b_b -= lr*db_b
            self.b = self.b_a + self.b_b 

data = pd.read_csv("breast-cancer-wisconsin.csv") 


total = len(data.values)
train = int(total * 0.8)
test = total - train
print train, 'for training', test, 'for validating'


X = data.values[0:train,1:10]


Y = data.values[:,10]

for i in range(len(Y)):
	if Y[i]==2:
		Y[i] = 1
	else:
		Y[i] = -1
positive = -1

y = Y[0:train]
ans = Y[train:]

svm = SVM(10, lr = 0.01)
svm.fit(X, y, iteration_times=2000)


correct = 0 
tp=0
fp=0
fn=0


test_set = data.values[train:total, :]
for i in range(len(test_set)):
    x = test_set[i, 1:10]
    y = test_set[i, 10]
    #print y
    pred = svm.predict(x)
    
    if pred == y:
        correct +=1 
    if y==positive:
        if pred==positive:
            tp+=1
            print 'tp', pred, y
        else:
            fn+=1
            print 'fn', pred, y
    else:
        if pred==positive:
            fp+=1
            print 'fp', pred, y

print 'tp, fn, fp', tp, fn, fp
print 'Precision', float(tp)/(tp+fp)
print 'Recall', float(tp)/(tp + fn)
print 'Correct/Test', correct, test