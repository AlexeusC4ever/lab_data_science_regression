# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science2')

import os
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import utils
import config as cfg


@click.command()
@click.argument('input_filepath_test', type=click.Path(exists=True))
@click.argument('output_filepath_test', type=click.Path())
@click.argument('best_model_path', type=click.Path())
def main(input_filepath_test, output_filepath_test,
         best_model_path):

    logger = logging.getLogger(__name__)
    logger.info('Make predictions')

    test = utils.load_as_pickle(input_filepath_test)
    model = utils.load_model_as_pickle(best_model_path)

    test_feat = test[cfg.FEATURES]
    test_id = test['Id']

    preds = model.predict(test_feat)

    preds = pd.DataFrame(preds, columns=[cfg.TARGET])
    preds['Id'] = test_id.astype(int)
    preds = preds[['Id', cfg.TARGET]]

    preds.to_csv(output_filepath_test, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
