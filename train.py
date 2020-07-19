from model import AutoEncoder
from visualizations import *

import argparse
import os


def run(args):
    # Create AutoEncoder
    autoencoder = AutoEncoder(args['input_shape'], args['z_dim'], args['c_dim'], learning_rate=args['learning_rate'])

    # train
    autoencoder.train(args['train_dir'], args['val_dir'], args['epochs'], args['batch_size'], args['output_dir'])

    # plot
    x = autoencoder.sample_data()
    plot_original(x, save_dir=args['output_dir'])
    plot_reconstruction(x, autoencoder, save_dir=args['output_dir'])
    plot_zvariation(x, autoencoder, save_dir=args['output_dir'])
    plot_cvariation(x, autoencoder, save_dir=args['output_dir'])
    plot_zsemireconstructed(x, autoencoder, save_dir=args['output_dir'])
    plot_csemireconstructed(x, autoencoder, save_dir=args['output_dir'])


if __name__ == '__main__':
    # Config (replace with arg parser)
    args = {
        'input_shape': (48, 48, 3),
        'z_dim': 99,
        'c_dim': 1,
        'beta': 1.,
        'learning_rate': 0.0005,
        'epochs': 12,
        'batch_size': 128,
        'train_dir': './celeba_data/train',
        'val_dir': './celeba_data/val',
        'output_dir': './results',
        'restore': False
    }

    # make dir
    os.makedirs(args['output_dir'], exist_ok=True)

    # run
    run(args)
