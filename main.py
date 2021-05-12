from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import pickle

import math
import os
import random
import sys
import time
from typing import List, Any

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import json
from matrics import compute_metrics
# from bid_metrics import compute_bid_metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_utils
import UPRR as CM
from click_model import ClickModel
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss
#
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# rank list size should be read from data
tf.app.flags.DEFINE_string("data_dir", "./data/MSLR10K_small/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./result/MSLR10K_train/", "Directory for training results and models")
tf.app.flags.DEFINE_string("test_dir", "./result/MSLR10K_test/", "Directory for output test results.")
tf.app.flags.DEFINE_string("bias_model_path", "",
                           "Use bias model from this path to initialize click model")
tf.app.flags.DEFINE_string("click_model_path", "",
                           "Use click model from this path to initialize ranker model")
tf.app.flags.DEFINE_string("restore_model_path", "",
                           "Restore/Test bias/click/ranker model from this path")
tf.app.flags.DEFINE_string("hparams", "", "Hyper-parameters for models.")

tf.app.flags.DEFINE_string("train_stage", 'ranker',
                           "traing stage.")

tf.app.flags.DEFINE_boolean("use_debias", False,
                            "use_debias")

tf.app.flags.DEFINE_integer("batch_size", 512,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("np_random_seed", 1385,
                            "random seed for numpy")
tf.app.flags.DEFINE_integer("tf_random_seed", 20933,
                            "random seed for tensorflow")
tf.app.flags.DEFINE_integer("emb_size", 10,
                            "Embedding to use during training.")
tf.app.flags.DEFINE_integer("train_list_cutoff", 10,
                            "The number of documents to consider in each list during training.")
tf.app.flags.DEFINE_integer("max_train_iteration", 0,
                            "Limit on the iterations of training (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for decoding data.")
tf.app.flags.DEFINE_boolean("decode_train", False,
                            "Set to True for decoding training data.")
tf.app.flags.DEFINE_boolean("decode_valid", False,
                            "Set to True for decoding valid data.")
# To be discarded.
tf.app.flags.DEFINE_boolean("feed_previous", False,
                            "Set to True for feed previous internal output for training.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Set to True for test program.")

FLAGS = tf.app.flags.FLAGS

tf.set_random_seed(FLAGS.tf_random_seed)
np.random.seed(FLAGS.np_random_seed)


def create_model(session, data_set, forward_only, ckpt = None):
    """Create model and initialize or load parameters in session."""
    print('create model', data_set.user_field_M, data_set.item_field_M)

    model = CM.UPRR(data_set.rank_list_size, data_set.user_field_M, data_set.item_field_M, FLAGS.emb_size,
                    FLAGS.batch_size, FLAGS.hparams,
                    forward_only, train_stage=FLAGS.train_stage, use_debias=FLAGS.use_debias)

    print(ckpt)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())

    if not ckpt:
        if FLAGS.train_stage == 'ranker':
            ckpt = tf.train.get_checkpoint_state(FLAGS.click_model_path)  # mslr
            print('Reloading partial parameters from', ckpt)
            click_variables_to_restore = [v for v in tf.global_variables() if
                                          v.name.split('/')[0] == 'ClickRelNet' or v.name.split('/')[
                                              0] == 'ClickExamNet' or v.name.split('/')[0] == 'Biasnet']
            saver = tf.train.Saver(click_variables_to_restore)
            saver.restore(session, ckpt.model_checkpoint_path)
        if FLAGS.train_stage == 'click' and FLAGS.use_debias:
            ckpt = tf.train.get_checkpoint_state(FLAGS.bias_model_path)
            print('Reloading partial parameters from', ckpt)
            bias_variables_to_restore = [v for v in tf.global_variables() if
                                         v.name.split('/')[0] == 'Biasnet']
            saver = tf.train.Saver(bias_variables_to_restore)
            saver.restore(session, ckpt.model_checkpoint_path)


    if not forward_only:
        if model.train_stage == 'click':
            model.separate_gradient_update()

        model.global_gradient_update()
        tf.summary.scalar('Loss', tf.reduce_mean(model.loss))
        tf.summary.scalar('Learning Rate', model.learning_rate)
        tf.summary.scalar('Final Loss', tf.reduce_mean(model.loss))
    model.summary = tf.summary.merge_all()

    uninitialized_vars = []
    for var in tf.global_variables():
        try:
            session.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    initialize_op = tf.variables_initializer(uninitialized_vars)
    session.run(initialize_op)

    return model


def train():
    # Prepare data.
    print("Reading data in %s" % FLAGS.data_dir)

    train_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff)
    valid_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff)
    # rand_valid_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff, ranker=['randRanker1', 'randRanker2'])
    print("Rank list size %d" % train_set.rank_list_size)
    click_model_1 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker1', set_name='test', eta=0.5)
    click_model_2 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker2', set_name='test', eta=0.5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Creating model...")
        # model = create_model(sess, train_set, None, ckpt = tf.train.get_checkpoint_state(check_point_dir))
        model = create_model(sess, train_set, False, ckpt=tf.train.get_checkpoint_state(FLAGS.restore_model_path))
        if FLAGS.train_stage == 'ranker':
            if FLAGS.use_debias:
                check_point_dir = FLAGS.train_dir + FLAGS.train_stage + "/lambdaloss/debias_"+ str(
                    FLAGS.hparams) + '_batch_'+ str(FLAGS.batch_size)
            else:
                check_point_dir = FLAGS.train_dir + FLAGS.train_stage + "/lambdaloss/withoutdebias_" + str(
                    FLAGS.hparams) + '_batch_' + str(FLAGS.batch_size)
        if FLAGS.train_stage== 'bias':
            check_point_dir = FLAGS.train_dir + FLAGS.train_stage + "/" + str(FLAGS.hparams) + '_batch_'+ str(FLAGS.batch_size)

        if FLAGS.train_stage == 'click':
            if FLAGS.use_debias:
                check_point_dir = FLAGS.train_dir + FLAGS.train_stage +"/debias_clip_"+str(model.hparams.bias_clip_min)+"_"+ str(
                    FLAGS.hparams) + '_batch_' + str(FLAGS.batch_size)
            else:
                check_point_dir = FLAGS.train_dir + FLAGS.train_stage + "/withoutdebias_" + str(
                    FLAGS.hparams)+ '_batch_' + str(FLAGS.batch_size)

        print(check_point_dir)

        # Create tensorboard summarizations.
        train_writer = tf.summary.FileWriter(FLAGS.train_dir + '/train_log',
                                             sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.train_dir + '/valid_log')
        if not os.path.exists(check_point_dir):
            print('mkdir', check_point_dir)
            os.makedirs(check_point_dir)
        log_file = open(check_point_dir + '/output.log', 'a')

        if FLAGS.train_stage == 'bias':
            # Training of bias model
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                input_feed = model.get_batch_for_bias(train_set)
                step_loss, _, summary, acc, l2_loss = model.bias_step(sess, input_feed, forward_only=False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                train_writer.add_summary(summary, current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    print("global step %d learning rate %.8f step-time %.2f loss %.4f acc %4f" % (
                    tf.convert_to_tensor(model.global_step).eval(),
                    tf.convert_to_tensor(model.learning_rate).eval(), step_time, loss, acc))

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    #     sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    # Validate model
                    it = 0
                    count_batch = 0.0
                    valid_loss = 0
                    acc_sum = 0
                    while it < valid_set.data.shape[0]:
                        input_feed = model.get_batch_for_bias(valid_set)
                        v_loss, summary, acc, propensity = model.bias_step(sess, input_feed, forward_only=True)
                        it += model.batch_size
                        valid_loss += v_loss
                        count_batch += 1.0
                        acc_sum += acc
                    valid_writer.add_summary(summary, current_step)
                    valid_loss /= count_batch
                    acc = acc_sum / count_batch
                    # valid_loss = math.exp(valid_loss) if valid_loss < 300 else float('inf')
                    print("  eval: loss %.4f acc %.4f" % (valid_loss, acc))

                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = check_point_dir + "/model.ckpt"
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                    if loss == float('inf'):
                        break

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

                    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                        break

        if FLAGS.train_stage == 'click':
            # Training of bias model
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            best_loss = None
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                input_feed, data = model.get_batch_for_click(train_set)
                step_loss, _, summary, auc, acc, click, rel_err = model.click_step(sess, input_feed, False, click_model_1, data, (current_step+1) % FLAGS.steps_per_checkpoint == 0)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                train_writer.add_summary(summary, current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:

                    # Print statistics for the previous epoch.
                    print("global step %d learning rate %.4f step-time %.2f loss "
                          "%.2f auc %.4f acc %.4f rel_err %.4f" % (tf.convert_to_tensor(model.global_step).eval(),
                                                                   tf.convert_to_tensor(model.learning_rate).eval(),
                                                                   step_time, loss,
                                                                   auc, acc, rel_err))
                    print("global step %d learning rate %.4f step-time %.2f loss "
                          "%.2f auc %.4f acc %.4f rel_err %.4f" % (tf.convert_to_tensor(model.global_step).eval(),
                                                                   tf.convert_to_tensor(model.learning_rate).eval(),
                                                                   step_time, loss,
                                                                   auc, acc, rel_err),file=log_file, flush=True)

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)

                    # Validate model
                    it, batch_num = 0, 0
                    bias_loss, rand_loss = 0, 0
                    bias_acc, rand_acc = 0, 0
                    bias_auc, rand_auc = 0, 0
                    bias_rel_error, rand_rel_error = 0, 0
                    print()
                    while batch_num < valid_set.data.shape[0]/FLAGS.batch_size:
                        input_feed1, data1 = model.get_batch_for_click_by_index(valid_set, it)
                        v_loss, summary, auc, acc, click, step_rel_error = model.click_step(sess, input_feed1, True, click_model_1, data1, True)
                        bias_loss += v_loss
                        bias_rel_error += step_rel_error
                        bias_acc += acc
                        bias_auc += auc
                        it += FLAGS.batch_size
                        batch_num += 1
                        # input_feed2, data2 = model.get_batch_for_click_by_index(rand_valid_set, it)
                        # v_loss, summary, auc, acc, click, step_rel_error = model.click_step(sess, input_feed2, True, click_model_1, data2, True)

                        # rand_loss += v_loss
                        # rand_rel_error += step_rel_error
                        # rand_auc += auc
                        # rand_acc += acc
                    valid_writer.add_summary(summary, current_step)
                    print("bias eval: loss %.4f auc %.4f acc %.4f rel_err %.4f" % (bias_loss/batch_num, bias_auc/batch_num, bias_acc, bias_rel_error))
                    # print("rand eval: loss %.4f auc %.4f acc %.4f rel_err %.4f" % (rand_loss/batch_num, rand_auc/batch_num, rand_acc, rand_rel_error))
                    print("bias eval: loss %.4f auc %.4f acc %.4f rel_err %.4f" % (bias_loss/batch_num, bias_auc/batch_num, bias_acc, bias_rel_error), file=log_file, flush=True)
                    # print("rand eval: loss %.4f auc %.4f acc %.4f rel_err %.4f" % (rand_loss/batch_num, rand_auc/batch_num, rand_acc, rand_rel_error), file=log_file, flush=True)

                    checkpoint_path = check_point_dir + "/model.ckpt"
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

                    if loss == float('inf'):
                        break

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

                    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                        break

        # Training of the ranker
        if FLAGS.train_stage == 'ranker':

            # Training of bias model
            step_time, loss = 0.0, 0.0
            current_step = 0
            while True:
                # Get a batch and make a step.
                start_time = time.time()
                input_feed, labels , clicks = model.get_batch_for_ranker(train_set)
                labels, scores, step_loss, summary, _ = model.ranker_step(sess, input_feed, labels, clicks, forward_only=False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1
                train_writer.add_summary(summary, current_step)

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if (current_step < 40) or (current_step % FLAGS.steps_per_checkpoint == 0):
                    # Print statistics for the previous epoch.

                    print("global step %d learning rate %.6f step-time %.2f loss "
                          "%.8f" % (tf.convert_to_tensor(model.global_step).eval(),
                                    tf.convert_to_tensor(model.learning_rate).eval(), step_time, loss))
                    print('on train set')

                    # Validate model
                    it = 0
                    count_batch = 0.0
                    all_labels, all_scores, all_rank, all_deltas = [], [], [], []
                    while it < valid_set.user_num:
                        input_feed, labels, clicks = model.get_batch_for_ranker_by_index(valid_set, it)
                        labels, scores = model.ranker_step(sess, input_feed, labels, clicks, forward_only=True)
                        it += 1
                        count_batch += 1.0
                        all_labels.extend(labels)
                        all_scores.extend(scores)
                        scores = scores[0]
                        labels = labels[0]
                        rank = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                        all_rank.append([labels[x] for x in rank])

                    print('on test set')
                    print(current_step)
                    compute_metrics(all_labels, all_scores, valid_set, None, log_file)

                    checkpoint_path = check_point_dir + "/model.ckpt"
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    if loss == float('inf'):
                        break

                    step_time, loss = 0.0, 0.0
                    sys.stdout.flush()

                    if FLAGS.max_train_iteration > 0 and current_step > FLAGS.max_train_iteration:
                        break


def decode(model_path, store_path=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        # Load test data.
        print("Reading data in %s" % FLAGS.data_dir)
        test_set = None
        if FLAGS.decode_train:
            test_set = data_utils.read_data(FLAGS.data_dir, 'train', FLAGS.train_list_cutoff)
        elif FLAGS.decode_valid:
            test_set = data_utils.read_data(FLAGS.data_dir, 'valid', FLAGS.train_list_cutoff)
        else:
            test_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff)
            # rand_test_set = data_utils.read_data(FLAGS.data_dir, 'test', FLAGS.train_list_cutoff, ranker=['randRanker1', 'randRanker2'])

        click_model_1 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker1', set_name='test', eta=0.5)
        click_model_2 = ClickModel(data_path=FLAGS.data_dir, ranker='ranker2', set_name='test', eta=0.5)

        # Create model and load parameters.
        # model = create_model(sess, test_set, False, ckpt = tf.train.get_checkpoint_state('./result/train/share_model/together_withoutbias_imbalancelearning_rate=1e-4_interval_200_exam_vs_rel_20_batch_2048'))
        print(model_path)
        model = create_model(sess, test_set, False, ckpt=tf.train.get_checkpoint_state(model_path))
        if store_path:
            store_path = store_path + model_path.split('/')[-1] + '_UPRR'


        # model = create_model(sess, test_set, False)
        # model.batch_size = 1  # We decode one sentence at a time.
        if FLAGS.train_stage == 'ranker':
            for i in range(1):
                all_labels, all_scores = [], []
                it_list = []
                for it in tqdm(range(int(test_set.user_num))):
                    input_feed, labels, clicks = model.get_batch_for_ranker_by_index(test_set, it)
                    labels, scores = model.ranker_step(sess, input_feed, labels, clicks, forward_only=True)
                    # labels, scores = model.ranker_step(sess, input_feed, labels, clicks, forward_only=True)
                    # test KM oracle
                    # labels, scores, click_table = model.KM_oracle(sess, input_feed, labels, clicks, True, click_model_1)
                    # test KM ranker
                    # labels, scores = model.KM_ranker(sess, input_feed, labels, clicks, forward_only=True)
                    #test direct ctr ranker
                    # labels, scores = model.direct_ctr_rank(sess, input_feed, labels, clicks, forward_only=True)
                    # labels, scores = model.rel_rank(sess, input_feed, labels, clicks, forward_only=True)
                    all_labels.extend(labels)
                    all_scores.extend(scores)

                print(len(it_list), it_list)
                print()
                compute_metrics(all_labels, all_scores, test_set, None, None)

        elif FLAGS.train_stage == 'click':
            it = 0
            count_batch = 0.0
            rand_valid_loss, bias_valid_loss = 0, 0
            all_acc = 0
            all_auc = 0
            rand_rel_error, bias_rel_error = 0, 0
            bias_label_list =[]
            bias_click_list = []
            rand_label_list =[]
            rand_click_list = []
            for i in tqdm(range(test_set.data.shape[0]//FLAGS.batch_size)):
                it = i * FLAGS.batch_size
                input_feed1, data1 = model.get_batch_for_click_by_index(test_set, it)
                v_loss, summary, auc, acc, click, step_rel_error = model.click_step(sess, input_feed1, True,
                                                                                    click_model_1, data1, False)
                bias_label_list.append(data1[:, 4])
                bias_click_list.append(click)
                bias_valid_loss += v_loss
                bias_rel_error += step_rel_error
                # input_feed2, data2 = model.get_batch_for_click_by_index(rand_test_set, it)
                # v_loss, summary, auc, acc, click, step_rel_error = model.click_step(sess, input_feed2, True,
                #                                                                     click_model_1, data2, False)
                # rand_label_list.append(data2[:, 4])
                # rand_click_list.append(click)
                # rand_valid_loss += v_loss
                # rand_rel_error += step_rel_error

                count_batch += 1.0

            bias_label_ary = np.concatenate(bias_label_list)
            bias_click_ary = np.concatenate(bias_click_list)
            # rand_label_ary = np.concatenate(rand_label_list)
            # rand_click_ary = np.concatenate(rand_click_list)
            bias_auc = roc_auc_score(bias_label_ary, bias_click_ary)
            bias_logloss = log_loss(bias_label_ary, bias_click_ary)
            # rand_auc = roc_auc_score(rand_label_ary, rand_click_ary)
            # rand_logloss = log_loss(rand_label_ary, rand_click_ary)

            bias_rel_error = bias_rel_error / count_batch
            bias_valid_loss = bias_valid_loss / count_batch
            # rand_rel_error = rand_rel_error / count_batch
            # rand_valid_loss = rand_valid_loss / count_batch
            print("bias test: loss %.2f auc %.4f rel_err %.4f log_loss %.4f" % (bias_valid_loss, bias_auc, bias_rel_error, bias_logloss))
            # print("rand test: loss %.2f auc %.4f rel_err %.4f log_loss %.4f" % (rand_valid_loss, rand_auc, rand_rel_error, rand_logloss))
        elif FLAGS.train_stage == 'bias':
            it = 0
            count_batch = 0.0
            valid_loss = 0
            acc_sum = 0
            prop = np.zeros((1, 10))
            while it < test_set.data.shape[0]:
                input_feed = model.get_batch_for_bias(test_set)
                v_loss, summary, acc, propensity = model.bias_step(sess, input_feed, forward_only=True)
                it += model.batch_size
                valid_loss += v_loss
                count_batch += 1.0
                acc_sum += acc
                prop += np.mean(propensity, axis=0)
            valid_loss /= count_batch
            acc = acc_sum / count_batch
            # valid_loss = math.exp(valid_loss) if valid_loss < 300 else float('inf')
            print("  eval: loss %.4f acc %.4f" % (valid_loss, acc))
            print('prop sum', prop/count_batch)




    return



def main(_):
    if FLAGS.decode:
        # choose the model to test
        decode(FLAGS.restore_model_path)
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
