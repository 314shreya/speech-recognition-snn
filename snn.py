import argparse
import os
import dataset
import network as net
import numpy as np

def run(path):
    audio_files = os.listdir(path)[:100]

    X_train, X_test, y_train, y_test = dataset.load(audio_files, path, test_size=0.2)

    print("Features shape: ", X_train.shape)
    print("Labels shape: ",y_train.shape)

    T = 1000
    dt = 0.1

    input_dimension = 200
    output_dimension = 10


    network = net.Network(0.02, 0.2, -65, 2, 30, T, dt, input_dimension, output_dimension)

    print("Network initialization done")

    batch_size, _ = X_train.shape
    output_spikes = np.zeros((batch_size, output_dimension))
    
    for example_index in range(batch_size):
        output = network.forward(X_train[example_index], y_train[example_index])
        output_spikes[example_index, :] = output 

    print(output_spikes)

    # Evaluate the network
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="./.data/recordings",
        type=str,
        help="Path to the downloaded data files",
    )
    args = parser.parse_args()

    run(args.path)
