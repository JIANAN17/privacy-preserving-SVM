from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5
import base64
import pandas as pd 
import math
import numpy as np
import time
# encryption

data = pd.read_csv("breast-cancer-wisconsin.csv") 

total = len(data.values)
train = int(total * 0.6)

# X = data.values[0:train,1:10] 
X = data.values[0:train,0:10] 
Y = data.values[:,10]

# X_a = np.random.randint(0,10,(546, 9))
X_a = np.random.randint(0,10,(409, 10))
X_b = X - X_a
Y_a = np.random.randint(0,10,(409,1))
Y_b = Y - Y_a
rsa = RSA.generate(1024)

plaintext = ""
for i in range(1,train):
	plaintext+=str(X_a[i])
	plaintext+=str(X_b[i])
	plaintext+=str(Y_a[i])
	plaintext+=str(Y_b[i])
	# plaintext.append(X_a[i])
	# plaintext.append(X_b[i])
	# plaintext.append(Y_a[i])
	# plaintext.append(Y_b[i])

privatekey = rsa.exportKey()
with open('privatekey.pem', 'w') as f:
    f.write(privatekey)

publickey = rsa.publickey().exportKey()
with open('publickey.pem', 'w') as f:
    f.write(publickey)

publickey = RSA.importKey(open('publickey.pem').read())
enc = PKCS1_v1_5.new(publickey)
encrypt_text = ""

time_on = time.time()
for i in range(0,len(plaintext),100):
        cont = plaintext[i:i+100]
        encrypt_text+=enc.encrypt(cont)
time_off = time.time()


with open("ciphertext", 'w') as f:
    f.write(base64.encodestring(encrypt_text))


time = time_off - time_on
print time




