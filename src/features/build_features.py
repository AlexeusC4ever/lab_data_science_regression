# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science2')

import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import src.config as cfg
import src.utils as utils


@click.command()
@click.argument('input_filepath_train_x', type=click.Path(exists=True))
@click.argument('input_filepath_train_y', type=click.Path())
@click.argument('input_filepath_val_x', type=click.Path(exists=True))
@click.argument('input_filepath_val_y', type=click.Path())
@click.argument('output_filepath_label_encoded_train_x', type=click.Path())
@click.argument('output_filepath_label_encoded_val_x', type=click.Path())
@click.argument('output_filepath_target_encoded_train_x', type=click.Path())
@click.argument('output_filepath_target_encoded_val_x', type=click.Path())
def main(input_filepath_train_x, input_filepath_train_y, input_filepath_val_x, input_filepath_val_y,
    output_filepath_label_encoded_train_x, output_filepath_label_encoded_val_x,
    output_filepath_target_encoded_train_x, output_filepath_target_encoded_val_x):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Building features')

    x_train = utils.load_as_pickle(input_filepath_train_x)
    x_val = utils.load_as_pickle(input_filepath_val_x)
    y_train = utils.load_as_pickle(input_filepath_train_y)
    y_val = utils.load_as_pickle(input_filepath_val_y)

    x_train_label_encoded, x_val_label_encoded = utils.label_encoder(cfg.CAT_COLS, x_train, x_val)
    
    targetEncoder = utils.TargetEncoder(cfg.CAT_COLS, [cfg.TARGET])
    x_train_target_encoded = targetEncoder.fit_transform(x_train, pd.DataFrame(y_train, columns=[cfg.TARGET]))
    x_val_target_encoded = targetEncoder.transform(x_val)

    utils.save_as_pickle(x_train_label_encoded, output_filepath_label_encoded_train_x)
    utils.save_as_pickle(x_val_label_encoded, output_filepath_label_encoded_val_x)
    utils.save_as_pickle(x_train_target_encoded, output_filepath_target_encoded_train_x)
    utils.save_as_pickle(x_val_target_encoded, output_filepath_target_encoded_val_x)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
