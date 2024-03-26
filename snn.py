import argparse
import os
import dataset

import network as net


def run(path):
    audio_files = os.listdir(path)[:100]

    labels, features = dataset.load(audio_files, path)

    print("Features shape: ", features.shape)
    print("Labels shape: ",labels.shape)

    # Convert the images to event images (Transform image to spikes)
    event_img = dataset.img_2_event_img(features, 20)
    print("Event image shape: ",event_img.shape)
    print("Event image length: ", len(event_img))

    # implement network
    T = 1000
    dt = 0.1

    network = net.Network(0.02, 0.2, -65, 2, 30, T, dt)

    # forward pass
    for i in range(len(event_img)):
        sample = event_img[i:i+1]  # Extract the i-th sample, keep the batch dimension
        output_spikes = network.forward(sample)
        print(f"Output spikes for sample {i}: {output_spikes}")


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
