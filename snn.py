import argparse
import os
import dataset
import network as net

def run(path):
    audio_files = os.listdir(path)[:100]

    X_train, X_test, y_train, y_test = dataset.load(audio_files, path, test_size=0.2)

    print("Features shape: ", X_train.shape)
    print("Labels shape: ",y_train.shape)

    # Convert the images to event images (Transform image to spikes)
    event_img = dataset.img_2_event_img(X_train, 20)
    print("Event image shape: ",event_img.shape)
    print("Event image length: ", len(event_img))

    T = 1000
    dt = 0.1

    input_dimension = 200
    hidden_dimension = 120
    output_dimension = 10


    network = net.Network(0.02, 0.2, -65, 2, 30, T, dt, input_dimension, output_dimension)

    print("Network initialization done")

    output = network.forward(event_img)

    print(output)

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
