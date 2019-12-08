'''Trains a simple deep NN for 1D PDE.

u_t + 0.5*sigma^2 u_xx + u_x mu + f = 0
u(T, x) = g(x)

X(n+1) - X(n) = mu dt + sigma dW(n)

U(n+1) - U(n) = -f(t(n), X(n), U(n), Z(n)) dt + Z(n) dW(n)

t in [0, T] Nt dt = T/Nt


Step 1: build dWn(Nt)
dWn is iid normal distribution ~ N(0, dt)

Step 2: build Xn(Nt) for given X0 using dWn
for n = 0:Nt-1
    X(n+1) - X(n) = mu(n) dt + sigma(n) dW(n)
end

Step 3: build Neural Network with

Input :        dW(1) ... dW(Nt-1) dW(Nt)
Lambda: X0      X(1) ...  X(Nt-1)  X(Nt)
Hidden:
...
Hidden: U0  Z0  Z(1) ...  Z(Nt-1)
Lambda:                            U(Nt)

Loss Function need X(Nt)

Step 4: build the loss function
for n = 0:Nt-1
    U(n+1) - U(n) = -f(t(n), X(n), U(n), Z(n)) dt + Z(n) dW(n)
end
Loss = ||U(Nt) - g(X(Nt))||


Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Dropout
from keras.optimizers import RMSprop

'''X(n+1) - X(n) = mu(n) dt + sigma(n) dW(n)'''
def build_X_from_dW(dW, mu=1, sigma=1, dt=0.1, X0=0.5):
    dX = mu*dt + sigma*dW
    X  = [X0,]
    for i in range(dW.shape[0]):
        X[i+1] = X[i] + dX[i]
    return X

'''U(n+1) - U(n) = -f(t(n), X(n), U(n), Z(n)) dt + Z(n) dW(n)'''
'''the first element of Z is U0'''
def build_U_from_Z(Z, f=1, dt=0.1, dW=[]):
    U  = Z[1]
    for i in range(len(dW)):
        U = U -f*dt + Z[i+1]*dW[i]
    return U

def func_g(x):
    return x**2

'''Loss = ||U(Nt) - g(X(Nt))||'''  '''get the last element of X'''
def custom_loss_wrapper(X): 
    def custom_loss(y_true,  y_pred): 
        return keras.mean(keras.square(func_g(X[-1])-y_pred))
    return custom_loss

batch_size = 10
epochs = 20

# create the random numbers ~ N(0, dt)
import numpy as np

# the shape of dW is 100, 10 which means T=1, Nt=10, dt=0.1
# dW ~ N(0, dt) where dt = 0.1
dW_train = np.random.randn(500, 10)*(0.1**0.5)
dW_test  = np.random.randn(100, 10)*(0.1**0.5)

print(dW_train.shape[0], 'train samples')
print(dW_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = []
y_test = []


dW = Input(shape=(10, )) 
X  = Lambda(build_X_from_dW)(dW)

hidden1 = Dense(128,  activation='relu')(X) 
hidden2 = Dropout(0.2)(hidden1)
hidden3 = Dense(128,  activation='relu')(hidden2) 
hidden4 = Dropout(0.2)(hidden3)
hidden5 = Dense(11,  activation='relu')(hidden4) 
out     = Lambda(build_U_from_Z, arguments={'dW':dW})(hidden5)
model   = Model(input=dW, output=out)

model.compile(loss=custom_loss_wrapper(X),  optimizer=RMSprop)
model.summary()

history = model.fit(dW_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(dW_test, y_test))
score = model.evaluate(dW_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
