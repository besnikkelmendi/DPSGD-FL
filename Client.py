# Python TCP Client A
import socket, pickle

global_model = []


# Network Class
class Client: 
    def __init__(self, port, host, clientModel):
        self.host = host
        self.port = port
        self.model = clientModel

    def start_client(self):

        BUFFER_SIZE = 120000 
  
        
        tcpClientA = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        tcpClientA.connect((self.host, self.port))


        
        
        tcpClientA.send(b'Hi!')
        while True:
            try: 
                data = tcpClientA.recv(BUFFER_SIZE)
            except:
                continue
            
            train_started = "Start training"

            if data == train_started.encode():

                print("Training is starting...")
                    
                # get the model from the server API
                paramenters = self.model.get_parameters()

                # train the model
                weights = self.model.fit(paramenters)
                # evaluate the model

                print("Evaluation is starting...")
                
                evaluation = self.model.evaluate(weights[0])
                print(evaluation)
                final_weights = self.model.get_final_parameters()
                # tcpClientA.send(str(final_weights[1]).encode('utf-8')) 
                tcpClientA.send(pickle.dumps(final_weights))
                
                break

            else:
                rcv_data = pickle.loads(data)
                self.model.set_parameters(rcv_data)
                continue
        tcpClientA.close() 



# client = Client(2004,socket.gethostname(),)
# client.start_client()