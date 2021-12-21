import Client
import socket, pickle
import tensorflow as tf
import ssl
import helper
import argparse
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
    VectorizedDPKerasSGDOptimizer,
)

HOST = socket.gethostname()
PORT = 2004
PRIVACY_LOSS = 0;

ssl._create_default_https_context = ssl._create_unverified_context

global_weights = []
# # Load and compile Keras model
# model =  helper.create_cnn_model()
# #model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])



# # Load MNIST dataset

# (x_train, y_train), (x_test, y_test) = helper.load(args.num_clients)[args.partition]

# if args.dpsgd and x_train.shape[0] % args.batch_size != 0:
#     drop_num = x_train.shape[0] % args.batch_size
#     x_train = x_train[:-drop_num]
#     y_train = y_train[:-drop_num]


# # Flattening
# x_train = x_train.reshape(60000,-1)
# x_test = x_test.reshape(10000, -1)

class MClient():
    def __init__(self, model, x_train, y_train, x_test, y_test, args):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.batch_size = args.batch_size
        self.local_epochs = args.local_epochs
        self.dpsgd = args.dpsgd

        if args.dpsgd:
            self.noise_multiplier = args.noise_multiplier
            if args.batch_size % args.microbatches != 0:
                raise ValueError(
                    "Number of microbatches should divide evenly batch_size"
                )
            optimizer = VectorizedDPKerasSGDOptimizer(
                l2_norm_clip=args.l2_norm_clip,
                noise_multiplier=args.noise_multiplier,
                num_microbatches=args.microbatches,
                learning_rate=args.learning_rate,
            )
            # Compute vector of per-example loss rather than its mean over a minibatch.
            loss = tf.keras.losses.CategoricalCrossentropy(
                from_logits=True, reduction=tf.losses.Reduction.NONE
            )
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.learning_rate)
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        # Compile model with Keras
        self. model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    
    # Compile model with Keras
    # self. model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    def get_parameters(self):
        # self.model.load_weights("client_weights.h5")
        return self.model.get_weights()

    def set_parameters(self, global_weights):
        # global_weights = global_weights
        self.model.set_weights(global_weights)
        

    def fit(self, parameters):
        global PRIVACY_LOSS
        if self.dpsgd:
            privacy_spent = helper.compute_epsilon(
                self.local_epochs,
                len(self.x_train),
                self.batch_size,
                self.noise_multiplier,
            )
            PRIVACY_LOSS += privacy_spent

        self.model.set_weights(parameters)
        self. model.fit(self.x_train, self.y_train, epochs=1, batch_size=self.batch_size, steps_per_epoch=250)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters):
        self. model.set_weights(parameters)
        loss, accuracy = self. model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

    def get_final_parameters(self):
        # modelName = "client_model_" + str(args.partition) + ".h5"
        # self.model.save(modelName)
        return self.model.get_weights()

    def get_client_parameters(self):
        """ Return all of the variables list """
        with self.graph.as_default():
            client_vars = sess.run(tf.trainable_variables())
        return client_vars
# Start Flower client
#fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=CifarClient())


# client = Client.Client(PORT,HOST,CifarClient())
# client.start_client()

def main(args) -> None:
    # Load Keras model
    model = helper.create_cnn_model()

    # Load a subset of MNIST to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = helper.load(args.num_clients)[args.partition]

    # drop samples to form exact batches for dpsgd
    # this is necessary since dpsgd is sensitive to uneven batches
    # due to microbatching
    if args.dpsgd and x_train.shape[0] % args.batch_size != 0:
        drop_num = x_train.shape[0] % args.batch_size
        x_train = x_train[:-drop_num]
        y_train = y_train[:-drop_num]

    # Start Flower client
    client = Client.Client(PORT,HOST,MClient(model, x_train, y_train, x_test, y_test, args))
    client.start_client()
    if args.dpsgd:
        print("Privacy Loss: ", PRIVACY_LOSS)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--num-clients",
        default=2,
        type=int,
        help="Total number of fl participants, requied to get correct partition",
    )
    parser.add_argument(
        "--partition",
        type=int,
        required=True,
        help="Data Partion to train on. Must be less than number of clients",
    )
    parser.add_argument(
        "--local-epochs",
        default=1,
        type=int,
        help="Total number of local epochs to train",
    )
    parser.add_argument("--batch-size", default=250, type=int, help="Batch size")
    parser.add_argument(
        "--learning-rate", default=0.15, type=float, help="Learning rate for training"
    )
    # DPSGD specific arguments
    parser.add_argument(
        "--dpsgd",
        default=False,
        type=bool,
        help="If True, train with DP-SGD. If False, " "train with vanilla SGD.",
    )
    parser.add_argument("--l2-norm-clip", default=1.0, type=float, help="Clipping norm")
    parser.add_argument(
        "--noise-multiplier",
        default=1.1,
        type=float,
        help="Ratio of the standard deviation to the clipping norm",
    )
    parser.add_argument(
        "--microbatches",
        default=25,
        type=int,
        help="Number of microbatches " "(must evenly divide batch_size)",
    )
    args = parser.parse_args()

    main(args)