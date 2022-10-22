# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science2')

import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
import src.config as cfg
import src.utils as utils


@click.command()
@click.argument('input_filepath_train', type=click.Path(exists=True))
@click.argument('output_filepath_train_x', type=click.Path())
@click.argument('output_filepath_train_y', type=click.Path())
@click.argument('output_filepath_val_x', type=click.Path())
@click.argument('output_filepath_val_y', type=click.Path())
def main(input_filepath_train, output_filepath_train_x, output_filepath_train_y, output_filepath_val_x, output_filepath_val_y):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Splitting data to train and val')

    train = utils.load_as_pickle(input_filepath_train)

    X, y = train[cfg.FEATURES], train[cfg.TARGET]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    utils.save_as_pickle(x_train, output_filepath_train_x)
    utils.save_as_pickle(y_train, output_filepath_train_y)
    utils.save_as_pickle(x_test, output_filepath_val_x)
    utils.save_as_pickle(y_test, output_filepath_val_y)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
