#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""execution script."""

import argparse
import os
import time
import torch

import matplotlib.pyplot as plt
import utils.exp_utils as utils
from detection import StatOnPredictions
# from plotting import plot_batch_prediction

# GG ??? Debug to remove
def printClass( obj ):
    print("__str__", obj.__str__ )
    dirs = dir(obj)
    print(dirs)
    print("__class__", obj.__class__ )
    for x, y in obj.__class__.items():
       print(x, y)
    print("__dict__", obj.values)

    attrs = vars(obj)
    print(obj.name, ', '.join("%s: %s" % item for item in attrs.items()))

def inference(cf, logger):
    """
    perform inference for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    # ???
    cf.shuffle = False
    net = model.net(cf, logger).cuda()
    batch_gen = data_loader.get_test_generator(cf, logger)
    data_set = batch_gen['test'].data_loader.getData()
    statObj = StatOnPredictions( cf, net, logger)
    # statObj.runStats( batch_gen, plot=False )
    statObj.runAndSave( batch_gen)
    o = statObj.loadPrediction()

    """
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    # GG : batch level, then data_loader, then [Single]thread
    data_set = batch_gen['test'].data_loader.getData()

    test_results_list = test_predictor.infer( batch_gen, return_results=True)
    test_evaluator.evaluate_predictions(data_set, test_results_list)
    # test_evaluator.score_test_df()
    """

if __name__ == '__main__':

    parser = argparse.ArgumentParser()  
    parser.add_argument('--mode', type=str,  default='inference',
                        help='Only: inference')
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non  existent.')
    parser.add_argument('--use_stored_settings', default=False, action='store_true',
                        help='load configs from existing exp_dir instead of source dir. always done for testing, '
                             'but can be set to true to do the same for training. useful in job scheduler environment, '
                             'where source code might change before the job actually runs.')
    parser.add_argument('--resume_to_checkpoint', type=str, default=None,
                        help='if resuming to checkpoint, the desired fold still needs to be parsed via --folds.')
    parser.add_argument('--exp_source', type=str, default='experiments/toy_exp',
                        help='specifies, from which source experiment to load configs and data_loader.')

    args = parser.parse_args()
    folds = args.folds
    resume_to_checkpoint = args.resume_to_checkpoint

    if args.mode == 'inference':
        server_env = False
        cf = utils.prep_exp(args.exp_source, args.exp_dir, server_env, is_training=False, use_stored_settings=True)
        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        # ??? plotter =  utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            try:
              logger = utils.get_logger(cf.fold_dir)
              cf.fold = fold
              inference(cf, logger)
            except FileNotFoundError:
              print("Stop at fold ", fold )

    else:
        raise RuntimeError('mode specified in args is not implemented...')

    # load raw predictions saved by predictor during testing, run aggregation algorithms and evaluation.
    """
    elif args.mode == 'analysis':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir)

        if cf.hold_out_test_set:
            cf.folds = args.folds
            predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
            results_list = predictor.load_saved_predictions(apply_wbc=True)
            utils.create_csv_output(results_list, cf, logger)

        else:
            if folds is None:
                folds = range(cf.n_cv_splits)
            for fold in folds:
                cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
                cf.fold = fold
                predictor = Predictor(cf, net=None, logger=logger, mode='analysis')
                results_list = predictor.load_saved_predictions(apply_wbc=True)
                logger.info('starting evaluation...')
                evaluator = Evaluator(cf, logger, mode='test')
                evaluator.evaluate_predictions(results_list)
                evaluator.score_test_df()

    # create experiment folder and copy scripts without starting job.
    # usefull for cloud deployment where configs might change before job actually runs.
    elif args.mode == 'create_exp':
        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, use_stored_settings=True)
        logger = utils.get_logger(cf.exp_dir)
        logger.info('created experiment directory at {}'.format(args.exp_dir))
    """

