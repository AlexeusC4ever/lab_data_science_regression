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
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import src.utils as utils


@click.command()
@click.argument('input_filepath_val_x', type=click.Path(exists=True))
@click.argument('input_filepath_val_y', type=click.Path())
@click.argument('input_filepath_label_encoded_val_x', type=click.Path())
@click.argument('input_filepath_target_encoded_val_x', type=click.Path())
@click.argument('catboost_model_path', type=click.Path())
@click.argument('rfr_label_model_path', type=click.Path())
@click.argument('rfr_target_model_path', type=click.Path())
@click.argument('best_model_path', type=click.Path())
def main(input_filepath_val_x, input_filepath_val_y,
         input_filepath_label_encoded_val_x, 
         input_filepath_target_encoded_val_x,
         catboost_model_path, 
         rfr_label_model_path,
         rfr_target_model_path,
         best_model_path):

    logger = logging.getLogger(__name__)
    logger.info('Make predictions')

    x_val = utils.load_as_pickle(input_filepath_val_x)
    y_val = utils.load_as_pickle(input_filepath_val_y)
    x_val_label = utils.load_as_pickle(input_filepath_label_encoded_val_x)
    x_val_target = utils.load_as_pickle(input_filepath_target_encoded_val_x)    

    best_score = 0
    best_model = None

    catboost_model = utils.load_model_as_pickle(catboost_model_path)
    RFR_model_label = utils.load_model_as_pickle(rfr_label_model_path)
    RFR_model_target = utils.load_model_as_pickle(rfr_target_model_path)

    model_names = ['catboost_model', 'RFR_model_label', 'RFR_model_target']
    models = [catboost_model, RFR_model_label, RFR_model_target]
    scores = []

    preds_catboost = catboost_model.predict(x_val)
    preds_RFR_label = RFR_model_label.predict(x_val_label)
    preds_RFR_target = RFR_model_target.predict(x_val_target)

    scores.append(MAE(y_val, preds_catboost))
    scores.append(MAE(y_val, preds_RFR_label))
    scores.append(MAE(y_val, preds_RFR_target))

    best_score_index = scores.index(min(scores))

    best_model = models[best_score_index]
    print('best_model:', model_names[best_score_index])

    utils.save_model_as_pickle(best_model, best_model_path)

    results = {}

    results['R2'] = [ r2_score(y_val, preds_catboost),
                      r2_score(y_val, preds_RFR_label),
                      r2_score(y_val, preds_RFR_target)]
    results['MSE'] = [MSE(y_val, preds_catboost),
                      MSE(y_val, preds_RFR_label),
                      MSE(y_val, preds_RFR_target)]
    results['MAE'] = [MAE(y_val, preds_catboost),
                      MAE(y_val, preds_RFR_label),
                      MAE(y_val, preds_RFR_target)]

    

    data = pd.DataFrame(data= np.c_[results['R2'], results['MSE'], results['MAE']],
     columns=['R2', 'MSE', 'MAE'], index=model_names)


    print(data)

    for col in results.keys():
        fig, axes = plt.subplots(dpi=100)   
        fig.subplots_adjust(bottom=0.2, top=0.95)
        axes.set_ylabel("Value of metric")
        data[col].plot.bar(ax=axes)
        fig.savefig(os.path.join("reports/figures/", "Metrics" + col + ".png"))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
