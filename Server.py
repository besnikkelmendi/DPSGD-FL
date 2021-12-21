from array import array
import socket, pickle 
import os, os.path
import threading
from threading import Thread 
from socketserver import ThreadingMixIn 
import random
import tensorflow as tf
import numpy as np
import helper


C = 1
MIN_NO_CLIENTS = 2

client_threads = []
client_ids = []
client_weights = []
server_weights_arr = []


model = helper.create_cnn_model()
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile("sgd", loss=loss, metrics=["accuracy"])

# Load MNIST dataset for evaluation
_, test = tf.keras.datasets.mnist.load_data()
x_test, y_test = test

# preprocessing
x_test, y_test = helper.preprocess(x_test, y_test)

def evaluate(parameters):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

def getServerWeights():
    if os.path.exists("main_model_weights.h5"):
        model.load_weights("main_model_weights.h5")
    return model.get_weights()

def createRandomSortedList(num, start = 1, end = 100):
        arr = []
        tmp = random.randint(start, end)
        
        for x in range(num):
            
            while tmp in arr:
                tmp = random.randint(start, end)
                
            arr.append(tmp)
            
        arr.sort()
        client_weights.clear()
        return arr
        
def FedAvgDepricated(c_weights, s_weights):
        sum = np.zeros((16,), dtype=int)
        print("Init shape: ", s_weights)
        for weights in c_weights:
            print("Weight: ", np.asarray(weights))
            sum = np.add(sum, np.asarray(weights))
        print("Sum: ", sum)
        avg_c_weights = []
        for weight in sum:
            avg_c_weights.append(float(float(weight)/float(len(c_weights))))
        
        s_weights[1] = np.asarray(avg_c_weights)
        print(s_weights)
        # ci
        return s_weights

def FedAvg(c_weights):

    for weights in c_weights:

        averaged_weights = model.get_weights()
        averaged_weights=[i * 0 for i in averaged_weights]

        client_weights = weights
        client_weights =  [i / (MIN_NO_CLIENTS * C) for i in client_weights]

    averaged_weights=[x + y for x, y in zip(averaged_weights,client_weights)]

    return averaged_weights


def Average():

    if len(client_weights) >= int(MIN_NO_CLIENTS*C):

        print("Averaging updates...\n")
        avg_weights = FedAvg(client_weights)
        
        client_weights.clear()

        print("Evaluation initiated...\n")
        evaluation = evaluate(avg_weights)
        print("Evaluation: \n",evaluation)
        
        model.set_weights(avg_weights)
        model.save("main_model_weights.h5")


class ClientThread(Thread): 
    
    def __init__(self,conn,ip,port): 
        Thread.__init__(self) 
        self.conn = conn
        self.ip = ip 
        self.port = port 

        print( "New client is available - " + ip + ":" + str(port))
        
 
    def run(self):

        trained = False
        
        print("Sending weights")
        model_string = pickle.dumps(model.get_weights())
        self.conn.send(model_string)
        while True:
                data = self.conn.recv(BUFFER_SIZE)

                if  not trained:
                    print("Initiated training for client " + str(self.ip) +":" + str(self.port))                             
                    self.conn.send(b'Start training') 
                    trained = True
                else:
                    c_weights = pickle.loads(data)
                    client_weights.append(c_weights) 
                    Average()
                    break

        self.conn.close()


TCP_IP = '0.0.0.0' 
TCP_PORT = 2004 
BUFFER_SIZE = 120000  

s_weights = getServerWeights()

server_weights_arr = getServerWeights()

tcpServer = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
tcpServer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
tcpServer.bind((TCP_IP, TCP_PORT)) 
threads = [] 
 
while True: 

    tcpServer.listen(4) 
    print( "MNIST Federated Learning Server: Waiting for clients...\n" )
    (conn, (ip,port)) = tcpServer.accept() 
    
    newthread = ClientThread(conn,ip,port) 
    newthread.start()
    # threads.append(newthread)

    # if len(threads) >= MIN_NO_CLIENTS:
        
    #     clients = createRandomSortedList(int(MIN_NO_CLIENTS*C), 0, len(threads)-2)
 
    #     for client in clients:
    #         threads[client].start()
    #         threads.pop(client)
            
    # client_ids.append(str(ip)+"|"+str(port))

    
        