# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science2')

import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import linear_model
import src.config as cfg
import src.utils as utils


@click.command()
@click.argument('input_filepath_train', type=click.Path(exists=True))
@click.argument('input_filepath_test', type=click.Path(exists=True))
@click.argument('output_filepath_train', type=click.Path())
@click.argument('output_filepath_test', type=click.Path())
def main(input_filepath_train, input_filepath_test, output_filepath_train, output_filepath_test):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df_train = pd.read_csv(input_filepath_train)
    df_test = pd.read_csv(input_filepath_test)

    imputer = SimpleImputer(strategy="most_frequent")
    df_train[cfg.CAT_COLS] = imputer.fit_transform(df_train[cfg.CAT_COLS])
    df_test[cfg.CAT_COLS] = imputer.fit_transform(df_test[cfg.CAT_COLS])

    imputer = SimpleImputer(strategy="median")  # median is more robust to outliers
    df_train[cfg.NUM_CALS] = imputer.fit_transform(df_train[cfg.NUM_CALS])
    df_test[cfg.NUM_CALS] = imputer.fit_transform(df_test[cfg.NUM_CALS])

    utils.save_as_pickle(df_train, output_filepath_train)
    utils.save_as_pickle(df_test, output_filepath_test)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
