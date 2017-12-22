from keras.models import load_model
from quiver_engine import server
import argparse


def main():
    """
    Analysis & Visualize Network Layers using quiver engine
    """
    DATA_DIR7 = '/data/quiver_imgs/'

    parser = argparse.ArgumentParser(description='Analysis & visualize network layers using quiver engine')
    parser.add_argument('-d', help='image directory',       dest='data_dir',            type=str, default=DATA_DIR7)
    parser.add_argument('-m', help='model file path',       dest='model_file',          type=str, default="model.h5")
    parser.add_argument('-p', help='server port',           dest='server_port',         type=int, default=4567)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    model = load_model(args.model_file)
    model.summary()

    server.launch(model, input_folder=args.data_dir, port=args.server_port)


if __name__ == '__main__':
    main()
