# -*- coding: utf-8 -*-
import sys
sys.path.insert(1, 'D:\git_repos\lab_data_science2')

import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error as MSE
from  catboost import CatBoostRegressor
import src.config as cfg
import src.utils as utils


@click.command()
@click.argument('input_filepath_train_x', type=click.Path(exists=True))
@click.argument('input_filepath_train_y', type=click.Path())
@click.argument('input_filepath_label_encoded_train_x', type=click.Path())
@click.argument('input_filepath_target_encoded_train_x', type=click.Path())
@click.argument('catboost_model_path', type=click.Path())
@click.argument('rfr_label_model_path', type=click.Path())
@click.argument('rfr_target_model_path', type=click.Path())
def main(input_filepath_train_x, input_filepath_train_y,
         input_filepath_label_encoded_train_x, 
         input_filepath_target_encoded_train_x,
         catboost_model_path, 
         rfr_label_model_path,
         rfr_target_model_path):

    logger = logging.getLogger(__name__)
    logger.info('Train models')

    x_train = utils.load_as_pickle(input_filepath_train_x)
    y_train = utils.load_as_pickle(input_filepath_train_y)
    x_train_label = utils.load_as_pickle(input_filepath_label_encoded_train_x)
    x_train_target = utils.load_as_pickle(input_filepath_target_encoded_train_x)

    best_score = 0
    best_model = None

    param_grid = {
        'model_size_reg': [0, 1],
        'n_estimators' : [500, 1000],
        'one_hot_max_size' : [0],
        'max_ctr_complexity': [1]
    }
    

    grid_catboost = GridSearchCV(CatBoostRegressor(cat_features=cfg.CAT_COLS, verbose=500), param_grid, cv=3, scoring=MSE, 
                        verbose=0)

    grid_catboost.fit(x_train, y_train)
    
    # catboost_model = CatBoostRegressor(cat_features=cat_idx, loss_function='MultiLogloss', verbose=100, **grid.best_params_)
    # catboost_model.fit(x_train, y_train)

    best_score = 0
    best_model = None
    
    param_grid = {
        'max_depth': [5, 10, 20],
        'n_estimators' : [500, 1000],
    }

    grid_RFR_label = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring=MSE, 
                        verbose=0)

    # print(x_train.shape)
    # print(x_train_label.shape)
    # print(y_train.shape)

    grid_RFR_label.fit(x_train_label, y_train)
    
    grid_RFR_target = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring=MSE, 
                        verbose=0)

    grid_RFR_target.fit(x_train_target, y_train)

    utils.save_model_as_pickle(grid_catboost, catboost_model_path)
    utils.save_model_as_pickle(grid_RFR_label, rfr_label_model_path)
    utils.save_model_as_pickle(grid_RFR_target, rfr_target_model_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
