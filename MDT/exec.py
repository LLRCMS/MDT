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

import utils.exp_utils as utils
from evaluator import Evaluator
from predictor import Predictor
from plotting import plot_batch_prediction
# Memory survey
import numpy as np
import linecache
import tracemalloc


def displayMonitoring(metrics, epoch):
#(ax1, metrics, exp_name, color_palette, epoch, figure_ix, separate_values_dict, do_validation):

    monitor_values_keys = metrics['train']['monitor_values'][1][0].keys()
    # separate_values = [v for fig_ix in separate_values_dict.values() for v in fig_ix]
    # if figure_ix == 0:
    #    plot_keys = [ii for ii in monitor_values_keys if ii not in separate_values]
    #    plot_keys += [k for k in metrics['train'].keys() if k != 'monitor_values']
    # else:
    #    plot_keys = separate_values_dict[figure_ix]
    plot_keys = [ii for ii in monitor_values_keys]
    plot_keys += [k for k in metrics['train'].keys() if k != 'monitor_values']

    x = np.arange(1, epoch + 1)
    for kix, pk in enumerate(plot_keys):
        if pk in metrics['train'].keys():
            y_train = metrics['train'][pk][1:]
            # if do_validation:
            #    y_val = metrics['val'][pk][1:]
            y_val = metrics['val'][pk][1:]
        else:
            y_train = [np.mean([er[pk] for er in metrics['train']['monitor_values'][e]]) for e in x]
            # if do_validation:
            #    y_val = [np.mean([er[pk] for er in metrics['val']['monitor_values'][e]]) for e in x]
            y_val = [np.mean([er[pk] for er in metrics['val']['monitor_values'][e]]) for e in x]
        # ax1.plot(x, y_train, label='train_{}'.format(pk), linestyle='--', color=color_palette[kix])
        print("train")
        print(x)
        print(y_train)
        print(y_val)
        # if do_validation:
        #    ax1.plot(x, y_val, label='val_{}'.format(pk), linestyle='-', color=color_palette[kix])
    # if epoch == 1:


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))



def train(logger):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """
    logger.info('performing training in {}D over fold {} on experiment {} with model {}'.format(
        cf.dim, cf.fold, cf.exp_dir, cf.model))

    # Trace memory
    tracemalloc.start()

    net = model.net(cf, logger).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=cf.learning_rate[0], weight_decay=cf.weight_decay)
    model_selector = utils.ModelSelector(cf, logger)
    train_evaluator = Evaluator(cf, logger, mode='train')
    val_evaluator = Evaluator(cf, logger, mode=cf.val_mode)

    starting_epoch = 1

    # prepare monitoring
    monitor_metrics, TrainingPlot = utils.prepare_monitoring(cf)

    if cf.resume_to_checkpoint:
        starting_epoch, monitor_metrics = utils.load_checkpoint(cf.resume_to_checkpoint, net, optimizer)
        logger.info('resumed to checkpoint {} at epoch {}'.format(cf.resume_to_checkpoint, starting_epoch))

    logger.info('loading dataset and initializing batch generators...')
    batch_gen = data_loader.get_train_generators(cf, logger)

    for epoch in range(starting_epoch, cf.num_epochs + 1):
        print("Memory allocated:", torch.cuda.memory_allocated() / 1024**2)
        snapshot = tracemalloc.take_snapshot()
        display_top(snapshot)

        logger.info('starting training epoch {}'.format(epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cf.learning_rate[epoch - 1]

        start_time = time.time()

        net.train()
        train_results_list = []

        for bix in range(cf.num_train_batches):
            batch = next(batch_gen['train'])
            tic_fw = time.time()
            results_dict = net.train_forward(batch)
            tic_bw = time.time()
            optimizer.zero_grad()
            results_dict['torch_loss'].backward()
            optimizer.step()
            logger.info('tr. batch {0}/{1} (ep. {2}) fw {3:.3f}s / bw {4:.3f}s / total {5:.3f}s || '
                        .format(bix + 1, cf.num_train_batches, epoch, tic_bw - tic_fw,
                                time.time() - tic_bw, time.time() - tic_fw) + results_dict['logger_string'])
            # train_results_list.append([results_dict['boxes'], batch['pid']])
            train_results_list.append([results_dict['boxes'], batch['pid']])
            monitor_metrics['train']['monitor_values'][epoch].append(results_dict['monitor_values'])

            # GG Memory leak
            # del batch
            train_time = time.time() - start_time

        _, monitor_metrics['train'] = train_evaluator.evaluate_predictions(train_results_list, monitor_metrics['train'])

        logger.info('starting validation in mode {}.'.format(cf.val_mode))
        with torch.no_grad():
            net.eval()
            if cf.do_validation:
                val_results_list = []
                val_predictor = Predictor(cf, net, logger, mode='val')
                for _ in range(batch_gen['n_val']):
                    batch = next(batch_gen[cf.val_mode])
                    if cf.val_mode == 'val_patient':
                        results_dict = val_predictor.predict_patient(batch)
                    elif cf.val_mode == 'val_sampling':
                        results_dict = net.train_forward(batch, is_validation=True)
                    val_results_list.append([results_dict['boxes'], batch['pid']])
                    monitor_metrics['val']['monitor_values'][epoch].append(results_dict['monitor_values'])
                    # GG Memory leak
                    # del batch

                _, monitor_metrics['val'] = val_evaluator.evaluate_predictions(val_results_list, monitor_metrics['val'])
                # print("monitor_metrics",monitor_metrics)
                model_selector.run_model_selection(net, optimizer, monitor_metrics, epoch)

            # update monitoring and prediction plots
            # GG Covid-19 TrainingPlot.update_and_save(monitor_metrics, epoch)
            displayMonitoring( monitor_metrics, epoch )
            epoch_time = time.time() - start_time
            logger.info('trained epoch {}: took {} sec. ({} train / {} val)'.format(
                epoch, epoch_time, train_time, epoch_time-train_time))
            batch = next(batch_gen['val_sampling'])
            results_dict = net.train_forward(batch, is_validation=True)
            logger.info('plotting predictions from validation sampling.')
            # GG Covid 19 plot_batch_prediction(batch, results_dict, cf)
            

def test(logger):
    """
    perform testing for a given fold (or hold out set). save stats in evaluator.
    """
    logger.info('starting testing model of fold {} in exp {}'.format(cf.fold, cf.exp_dir))
    net = model.net(cf, logger).cuda()
    test_predictor = Predictor(cf, net, logger, mode='test')
    test_evaluator = Evaluator(cf, logger, mode='test')
    batch_gen = data_loader.get_test_generator(cf, logger)
    test_results_list = test_predictor.predict_test_set(batch_gen, return_results=True)
    test_evaluator.evaluate_predictions(test_results_list)
    test_evaluator.score_test_df()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str,  default='train_test',
                        help='one out of: train / test / train_test / analysis / create_exp')
    parser.add_argument('--folds', nargs='+', type=int, default=None,
                        help='None runs over all folds in CV. otherwise specify list of folds.')
    parser.add_argument('--exp_dir', type=str, default='/path/to/experiment/directory',
                        help='path to experiment dir. will be created if non existent.')
    parser.add_argument('--server_env', default=False, action='store_true',
                        help='change IO settings to deploy models on a cluster.')
    parser.add_argument('--slurm_job_id', type=str, default=None, help='job scheduler info')
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

    if args.mode == 'train' or args.mode == 'train_test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, args.use_stored_settings)
        cf.slurm_job_id = args.slurm_job_id
        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            cf.fold = fold
            cf.resume_to_checkpoint = resume_to_checkpoint
            if not os.path.exists(cf.fold_dir):
                os.mkdir(cf.fold_dir)
            logger = utils.get_logger(cf.fold_dir)
            train(logger)
            cf.resume_to_checkpoint = None
            if args.mode == 'train_test':
                test(logger)

    elif args.mode == 'test':

        cf = utils.prep_exp(args.exp_source, args.exp_dir, args.server_env, is_training=False, use_stored_settings=True)
        cf.slurm_job_id = args.slurm_job_id
        model = utils.import_module('model', cf.model_path)
        data_loader = utils.import_module('dl', os.path.join(args.exp_source, 'data_loader.py'))
        if folds is None:
            folds = range(cf.n_cv_splits)

        for fold in folds:
            cf.fold_dir = os.path.join(cf.exp_dir, 'fold_{}'.format(fold))
            try:
              logger = utils.get_logger(cf.fold_dir)
              cf.fold = fold
              test(logger)
            except FileNotFoundError:
              print("Stop at fold ", fold )

    # load raw predictions saved by predictor during testing, run aggregation algorithms and evaluation.
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

    else:
        raise RuntimeError('mode specified in args is not implemented...')
