from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import sys
import time
import numpy as np
from typing import Any

import data_utils
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
# We disable pylint because we need python3 compatibility.
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import six

from sklearn.metrics import roc_auc_score
import copy
import itertools
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

import networkx as nx
import networkx.algorithms.matching as matching
from scipy import sparse




def imerge(a, b):
    for i, j in itertools.izip(a, b):
        yield i
        yield j


def sigmoid_prob(logits):
    return tf.sigmoid(logits - tf.reduce_mean(logits, -1, keep_dims=True))


def network(input_data, layer_sizes, scope, is_training, reuse=False, assign_vars=None):
    with variable_scope.variable_scope(scope, reuse=reuse):
        output_data = input_data
        output_sizes = layer_sizes
        current_size = input_data.get_shape().as_list()[-1]
        var_list=[]
        for i in range(len(output_sizes)):
            if assign_vars is None:
                expand_W = variable_scope.get_variable("expand_W_%d" % i, [current_size, output_sizes[i]],
                                                   initializer=tf.contrib.layers.xavier_initializer())
            else:
                expand_W = variable_scope.get_variable("expand_W_%d" % i, initializer=assign_vars[2*i+0])

            var_list.append(expand_W)
            if assign_vars is None:
                expand_b = variable_scope.get_variable("expand_b_%d" % i, [output_sizes[i]])
            else:
                expand_b = variable_scope.get_variable("expand_b_%d" % i, initializer=assign_vars[2*i+1])
            var_list.append(expand_b)
            output_data = tf.nn.bias_add(tf.matmul(output_data, expand_W), expand_b)
            # output_data = tf.layers.batch_normalization(output_data, training=is_training)
            output_data = tf.nn.relu(output_data)
            current_size = output_sizes[i]
        return output_data, var_list


def KM(adj_matrix, N = 10):
    G = nx.algorithms.bipartite.matrix.from_biadjacency_matrix(sparse.csr_matrix(adj_matrix))
    M = matching.max_weight_matching(G)
    # print('KM', M)
    poss = np.zeros([N], dtype = np.int32)
    for i,j in M:
        if i < N:
            poss[i] = j-N
        else:
            poss[j] = i-N
    # print(poss)
    return poss



class UPRR(object):

    def __init__(self, rank_list_size, user_field_M, item_field_M, embed_size, batch_size, hparam_str,
                 forward_only=False, sparse_input=False, train_stage='bias', use_debias=False):
        """Create the model.

		Args:
			rank_list_size: size of the ranklist.
			batch_size: the size of the batches used during training;
						the model construction is not independent of batch_size, so it cannot be
						changed after initialization.
			embed_size: the size of the input feature vectors.
			forward_only: if set, we do not construct the backward pass in the model.
		"""
        self.hparams = tf.contrib.training.HParams(
            learning_rate=8e-4,  # Learning rate.
            learning_rate_decay_factor=0.96,  # Learning rate decays by this much.
            max_gradient_norm=0.0,  # Clip gradients to this norm.
            l2_loss=0.0,  # Set strength for L2 regularization.
            relevance_category_num=5,  # Select the number of relevance category
            # hidden_layer_sizes=[1024, 512, 50],  # for the share hidden layer
            update_target_ranker_interval=1,
            bias_hidden_layer_sizes=[1024, 1024, 512, 50],
            click_hidden_layer_sizes=[1024, 1024, 512, 50],
            ranker_hidden_layer_sizes=[1024, 1024, 512, 50],
            pair_each_query=40,
            bias_clip_max = 1,
            bias_clip_min =0.005,
        )
        print(hparam_str)
        self.hparams.parse(hparam_str)
        self.learning_rate = tf.Variable(float(self.hparams.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * self.hparams.learning_rate_decay_factor)
        self.start_index = 0
        self.count = 1
        self.rank_list_size = rank_list_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.train_stage = train_stage
        self.use_debias = use_debias
        self.bias_model_step = tf.Variable(0, trainable=False)
        self.click_model_step = tf.Variable(0, trainable=False)
        self.ranker_model_step = tf.Variable(0, trainable=False)
        self.forward_only = forward_only
        self.data_step = 0
        self.data_permutation = None
        self.propensity = []

        self.update_target_ranker_interval = self.hparams.update_target_ranker_interval

        #####not use now
        self.update_interval = 200
        self.exam_vs_rel = 20


        # Feeds for inputs.
        if sparse_input:
            self.user_ids = tf.placeholder(tf.int32, shape=[None], name='user_id')
            self.item_ids = tf.placeholder(tf.int32, shape=[None], name='item_id')
        else:
            self.user_context_feature = tf.placeholder(tf.float32, shape=[None, user_field_M], name="user_context_feature")
            self.item_context_feature = tf.placeholder(tf.float32, shape=[None, user_field_M], name="item_context_feature")
            self.origin_item_feature = tf.placeholder(tf.float32, shape=[None, item_field_M], name="origin_item_feature")
        self.pos = tf.placeholder(tf.int32, shape=[None], name='pos')
        self.target_clicks = tf.placeholder(tf.float32, shape=[None], name='target_clicks')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.pos_neg_ratio = tf.placeholder(tf.float32, name='pos_neg_ratio')

        if self.train_stage == 'ranker':
            if not forward_only:
                if sparse_input:
                    self.pos_user_ids = tf.placeholder(tf.int32, shape=[None], name='pos_user_id')
                    self.pos_item_ids = tf.placeholder(tf.int32, shape=[None], name='pos_item_id')
                    self.neg_user_ids = tf.placeholder(tf.int32, shape=[None], name='neg_user_id')
                    self.neg_item_ids = tf.placeholder(tf.int32, shape=[None], name='neg_item_id')
                else:
                    self.pos_user_feature = tf.placeholder(tf.float32, shape=[None, user_field_M],
                                                           name="pos_user_feature")
                    self.pos_item_feature = tf.placeholder(tf.float32, shape=[None, item_field_M],
                                                           name="pos_item_feature")
                    self.neg_user_feature = tf.placeholder(tf.float32, shape=[None, user_field_M],
                                                           name="neg_user_feature")
                    self.neg_item_feature = tf.placeholder(tf.float32, shape=[None, item_field_M],
                                                           name="neg_item_feature")
                self.deltaR = tf.placeholder(tf.float32, shape=[None], name="delta_revenue")
                self.pair_label = tf.placeholder(tf.float32, shape=[None], name="pair_label")
            else:
                pass

        if use_debias:
            self.propensity, self.bias_loss = self.BaisNet()
        self.exam_prob, self.rel_prob, self.click, self.click_loss, self.click_acc = self.ClickNet(use_debias=use_debias)
        self.score = self.RankNet(self.user_context_feature, self.origin_item_feature, scope ='target_ranker')
        _ = self.RankNet(self.user_context_feature, self.origin_item_feature, scope='current_ranker')


        if not forward_only:
            # Gradients and SGD update operation for training the model.
            if self.train_stage == 'click':
                self.loss = self.click_loss
                self.global_step = self.click_model_step
            if self.train_stage == 'bias':
                self.loss = self.bias_loss
                self.global_step = self.bias_model_step
            if self.train_stage == 'ranker':
                self.pos_score = self.RankNet(self.pos_user_feature, self.pos_item_feature, scope = 'current_ranker')
                self.neg_score = self.RankNet(self.neg_user_feature, self.neg_item_feature, scope = 'current_ranker')
                self.ranker_loss = tf.reduce_mean(
                    self.deltaR * self.pair_label * tf.nn.sigmoid_cross_entropy_with_logits(logits=(self.pos_score - self.neg_score),
                                                                          labels=tf.ones_like(self.pair_label)))
                self.loss = self.ranker_loss
                self.global_step = self.ranker_model_step

        if not forward_only:
            # Select optimizer
            self.optimizer_func = tf.train.AdamOptimizer
        self.copy_model_parameters('current_ranker', 'target_ranker')

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        print(' init finish')

    def copy_model_parameters(self, scope1, scope2):
        """
         Copies the model parameters of one estimator to another.
    #
    #     Args:
    #       sess: Tensorflow session instance
    #       ranker1: Estimator to copy the paramters from
    #       ranker2: Estimator to copy the parameters to
    #     """
        r1_params = [t for t in tf.trainable_variables() if t.name.startswith(scope1)]
        r1_params = sorted(r1_params, key=lambda v: v.name)
        r2_params = [t for t in tf.trainable_variables() if t.name.startswith(scope2)]
        r2_params = sorted(r2_params, key=lambda v: v.name)

        self.copy_ops = []
        for r1_v, r2_v in zip(r1_params, r2_params):
            op = r2_v.assign(r1_v)
            self.copy_ops.append(op)


    def separate_gradient_update(self):
        opt = self.optimizer_func(self.hparams.learning_rate)
        click_exam_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ClickExamNet") + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "share_layer")
        click_rel_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "ClickRelNet")
        self.click_exam_update = opt.minimize(self.loss, var_list=click_exam_params, global_step=self.global_step)
        self.click_rel_update = opt.minimize(self.loss, var_list=click_rel_params, global_step=self.global_step)
        self.pretrain_exam_update = opt.minimize(-tf.reduce_mean((self.target_clicks * tf.log(self.exam_prob) + (1 - self.target_clicks) * tf.log(1 - self.exam_prob))), var_list=click_exam_params, global_step=self.global_step)


    def global_gradient_update(self):
        print('bulid gradient')
        self.l2_loss = tf.Variable(0.0, trainable=False)
        params = tf.trainable_variables()
        opt = self.optimizer_func(self.hparams.learning_rate)
        # opt = tf.train.GradientDescentOptimizer(self.ranker_learning_rate)
        self.gradients = tf.gradients(self.loss, params)
        if self.hparams.max_gradient_norm > 0:
            self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients, self.hparams.max_gradient_norm)
        # self.updates = opt.apply_gradients(zip(self.clipped_gradients, params), global_step=self.global_step)
        else:
            self.norm = tf.global_norm(self.gradients)
        # self.updates = opt.apply_gradients(zip(self.gradients, params), global_step=self.global_step)
        self.updates = opt.minimize(self.loss, global_step=self.global_step)


    def ClickNet(self, scope=None, use_debias=False):

        with variable_scope.variable_scope(scope or "ClickRelNet"):
            rel_input = tf.concat(axis = 1, values = [self.user_context_feature, self.origin_item_feature])
            hidden, self.click_var_list =  network(rel_input, self.hparams.click_hidden_layer_sizes, 'share_layer', is_training=self.is_training,
                                            reuse=False)
            self.click_last_W = variable_scope.get_variable("ClickRel_last_W",
                                                 [self.hparams.click_hidden_layer_sizes[-1], 1],
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.click_last_b = variable_scope.get_variable("ClickRel_last_b", [1])
            rel_logit = tf.squeeze(
                tf.nn.dropout(tf.nn.bias_add(tf.matmul(hidden, self.click_last_W), self.click_last_b), keep_prob=self.keep_prob))
        with variable_scope.variable_scope(scope or "ClickExamNet"):
            exam_input = self.item_context_feature
            hidden , _= network(exam_input, self.hparams.click_hidden_layer_sizes, 'ClickExamNet', is_training=self.is_training,
                        reuse=False)
            last_W = variable_scope.get_variable("ClickExam_last_W",
                                                 [self.hparams.click_hidden_layer_sizes[-1], self.rank_list_size],
                                                 initializer=tf.contrib.layers.xavier_initializer())
            last_b = variable_scope.get_variable("ClickExam_last_b", [self.rank_list_size])
            exam_logits = tf.nn.dropout(tf.nn.bias_add(tf.matmul(hidden, last_W), last_b), keep_prob=self.keep_prob)

            pos_one_hot = tf.one_hot(self.pos, depth=self.rank_list_size)
            exam_logit = tf.reduce_sum(exam_logits * pos_one_hot, axis=-1)


        label = self.target_clicks
        rel_prob = tf.nn.sigmoid(rel_logit)
        exam_prob = tf.nn.sigmoid(exam_logit)
        y_prob = exam_prob * rel_prob
        self.y_prob = tf.reduce_sum(tf.cast((y_prob > 0.5), dtype=tf.int32))

        click_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast((y_prob > 0.5), dtype=tf.int32), tf.cast(label, dtype=tf.int32)),
                    dtype=tf.float32))
        if use_debias:
            propensity = tf.clip_by_value(self.propensity, self.hparams.bias_clip_min, self.hparams.bias_clip_max)
            debias_term = tf.reduce_sum(
                tf.stop_gradient(propensity) * tf.one_hot(self.pos, depth=self.rank_list_size), axis=-1)
            print('debias', self.propensity.shape, tf.one_hot(self.pos, depth=self.rank_list_size).shape,
                  debias_term.shape)
            click_loss = -tf.reduce_mean((label * tf.log(y_prob) + (1 - label) * tf.log(1 - y_prob)) / (debias_term + 1e-7))
        else:
            click_loss = -tf.reduce_mean((label * tf.log(y_prob) + (1 - label) * tf.log(1 - y_prob)))
        return exam_prob, rel_prob, y_prob, click_loss, click_acc

    def BaisNet(self, scope=None):
        input = tf.concat(
            [self.user_context_feature, self.origin_item_feature], axis=1)
        label = self.pos
        hidden, _ = network(input, self.hparams.bias_hidden_layer_sizes, 'Biasnet', is_training=self.is_training, reuse=False)
        last_W = variable_scope.get_variable("Bias_last_W", [self.hparams.bias_hidden_layer_sizes[-1], self.rank_list_size],
                                             initializer=tf.contrib.layers.xavier_initializer())
        last_b = variable_scope.get_variable("Bias_last_b", [self.rank_list_size])
        logit = tf.squeeze(tf.nn.bias_add(tf.matmul(hidden, last_W), last_b))
        propensity = tf.nn.softmax(logit)
        self.bias_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.cast(tf.argmax(propensity, axis=1), dtype=tf.int32), label), dtype=tf.float32), axis=-1)
        bias_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label), axis=-1)
        return propensity, bias_loss

    def RankNet(self, user_feature, item_feature, scope='Ranknet'):
        embeddings = tf.concat(axis=1, values=[user_feature, item_feature])
        hidden, _ = network(embeddings, self.hparams.ranker_hidden_layer_sizes, scope, is_training=self.is_training, reuse=tf.AUTO_REUSE, assign_vars=self.click_var_list)
        with variable_scope.variable_scope('Ranknet', reuse=tf.AUTO_REUSE):
            last_W = variable_scope.get_variable("Rank_last_W",
                                                 initializer=self.click_last_W)
            last_b = variable_scope.get_variable("Rank_last_b", initializer = self.click_last_b)
        output_score = tf.squeeze(tf.nn.bias_add(tf.matmul(hidden, last_W), last_b))
        return output_score

    def bias_step(self, session, input_feed, forward_only):
        # Output feed: depends on whether we do a backward step or not.
        # print(input_feed)
        input_feed[self.is_training.name] = not forward_only
        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.loss,  # Loss for this batch.
                           self.summary,  # Summarize statistics.
                           self.bias_acc,
                           self.l2_loss,
                           self.propensity,
                           ]
        else:
            output_feed = [self.loss,  # Loss for this batch.
                           self.summary,  # Summarize statistics
                           self.bias_acc,
                           self.l2_loss,
                           self.propensity
                           ]
        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], None, outputs[2], outputs[3], outputs[4]  # loss, no outputs, summary, acc.
        else:
            return outputs[0], outputs[1], outputs[2], outputs[4]  # loss, outputs, summary, acc.

    def click_step(self, session, input_feed, forward_only, click_model, data, rel_err):
        # Output feed: depends on whether we do a backward step or not.
        # input_feed[self.keep_prob.name] = 1.0 if forward_only else 0.9
        input_feed[self.keep_prob.name] = 1.0
        input_feed[self.is_training.name] = not forward_only

        self.updates = [self.click_exam_update, self.click_rel_update]
        if not forward_only:
            output_feed = [self.updates,  # Update Op that does SGD.
                           self.click_loss,  # Loss for this batch.
                           self.summary,  # Summarize statistics.
                           # self.propensity,
                           self.click_acc,
                           self.propensity,
                           self.exam_prob,
                           self.rel_prob,
                           self.click

                           ]
        else:
            output_feed = [self.click_loss,  # Loss for this batch.
                           self.summary,  # Summarize statistics.
                           # self.propensity,
                           self.click_acc,
                           self.exam_prob,
                           self.propensity,
                           self.rel_prob,
                           self.click
                           ]

        outputs = session.run(output_feed, input_feed)
        if rel_err:
            item_num = input_feed[self.user_context_feature.name].shape[0]
            pred_click_per_pos = session.run(self.click, feed_dict = {self.user_context_feature.name: np.tile(input_feed[self.user_context_feature.name], (1, 10)).reshape(item_num*10, -1),
                                                                      self.item_context_feature.name:np.tile(input_feed[self.item_context_feature.name], (1, 10)).reshape(item_num*10, -1),
                                                                      self.origin_item_feature.name:np.tile(input_feed[self.origin_item_feature.name], (1, 10)).reshape(item_num*10, -1),
                                                                      self.pos.name:np.tile(np.arange(0, 10), item_num),
                                                                      self.keep_prob.name: 1.0,
                                                                      self.is_training: False})
            true_click_per_pos, _, _ = click_model.get_click_prob(np.tile(input_feed[self.origin_item_feature.name], (1, 10)).reshape(item_num*10, -1) , np.tile(np.arange(0, 10), item_num),
                                                            np.tile(data[:, 5].reshape(-1, 1), (1, 10)).reshape(-1, 1))

            rel_err = np.mean(np.abs(1 - np.tile(1/np.log(np.arange(2, 12)), item_num)*pred_click_per_pos/(true_click_per_pos+1e-30)))


        auc_score = roc_auc_score(input_feed[self.target_clicks.name], outputs[-1])

        if not forward_only:
            return outputs[1], None, outputs[2], auc_score, outputs[3], outputs[-1], rel_err  # loss, no outputs, summary.
        else:
            return outputs[0], outputs[1], auc_score, outputs[2], outputs[-1], rel_err  # loss, outputs, summary.



    def direct_ctr_rank(self, session, input_feed, labels, clicks, forward_only):
        input_feed[self.keep_prob.name] = 1.0
        input_feed[self.pos.name] = np.ones(shape=[input_feed[self.user_context_feature.name].shape[0]], dtype=np.int)
        flatten_scores = np.reshape(session.run(self.click, input_feed), [-1])
        scores = []
        start = 0
        for sublabels in labels:
            seqlen = len(sublabels)
            scores.append(flatten_scores[start:start + seqlen].tolist())
            start = start + seqlen
        return labels, scores

    def rel_rank(self, session, input_feed, labels, clicks, forward_only):
        input_feed[self.keep_prob.name] = 1.0
        flatten_scores = np.reshape(session.run(self.rel_prob, input_feed), [-1])
        scores = []
        start = 0
        for sublabels in labels:
            seqlen = len(sublabels)
            scores.append(flatten_scores[start:start + seqlen].tolist())
            start = start + seqlen
        return labels, scores

    def ranker_step(self, session, input_feed, labels, clicks, forward_only):
        if self.global_step.eval() % self.update_target_ranker_interval == 0:
            session.run(self.copy_ops)
        flatten_scores = np.reshape(session.run(self.score, input_feed), [-1])

        flatten_clicks = np.concatenate(clicks, axis=0)

        if not forward_only:
            scores = []
            current_poss = []
            origin_poss = []
            start = 0
            rand_pairs = []
            for index, sublabels in enumerate(labels):
                seqlen = len(sublabels)
                in_query_scores = flatten_scores[start:start + seqlen].tolist()
                scores.append(in_query_scores)
                sorted_score = sorted(in_query_scores, reverse=True)
                map = {v: i for i, v in enumerate(sorted_score)}
                current_poss.extend([map[v] for v in scores[-1]])
                origin_poss.extend(np.arange(seqlen))
                clicked_items = np.nonzero(clicks[index])[0]
                for ii in range(self.hparams.pair_each_query):
                    if clicked_items.shape[0]:
                        random_pair = np.zeros([2], dtype=np.int)
                        random_pair[0] = np.random.choice(clicked_items)
                        random_pair[1] = np.random.choice(seqlen)
                    else:
                        random_pair = np.random.choice(seqlen, 2)
                    rand_pairs.append(start + random_pair)

                start = start + seqlen
            rand_pairs = np.array(rand_pairs)
            origin_click_prob = np.reshape(session.run(self.click, feed_dict={
                self.user_context_feature.name: input_feed[self.user_context_feature.name],
                self.item_context_feature.name: input_feed[self.item_context_feature.name],
                self.origin_item_feature.name: input_feed[self.origin_item_feature.name],
                self.pos.name: np.array(origin_poss),
                self.keep_prob.name: 1.0}), [-1])

            click_i = flatten_clicks[rand_pairs[:, 0]]
            click_i_old = session.run(self.click, feed_dict={
                self.user_context_feature.name: input_feed[self.user_context_feature.name][rand_pairs[:, 0]],
                self.item_context_feature.name: input_feed[self.item_context_feature.name][rand_pairs[:, 0]],
                self.origin_item_feature.name: input_feed[self.origin_item_feature.name][rand_pairs[:, 0]],
                self.pos.name: np.array(current_poss)[rand_pairs[:, 0]],
                self.keep_prob.name: 1.0})
            click_i_new = session.run(self.click,
                                      feed_dict={
                                          self.user_context_feature.name:
                                              input_feed[self.user_context_feature.name][rand_pairs[:, 0]],
                                          self.origin_item_feature.name: input_feed[self.origin_item_feature.name][
                                              rand_pairs[:, 0]],
                                          self.item_context_feature.name:
                                              input_feed[self.item_context_feature.name][
                                                  rand_pairs[:, 0]],
                                          self.pos.name: np.array(current_poss)[rand_pairs[:, 1]],
                                          self.keep_prob.name: 1.0})
            click_i_log = origin_click_prob[rand_pairs[:, 0]]

            click_j = flatten_clicks[rand_pairs[:, 1]]
            click_j_old = session.run(self.click, feed_dict={
                self.user_context_feature.name: input_feed[self.user_context_feature.name][rand_pairs[:, 1]],
                self.item_context_feature.name: input_feed[self.item_context_feature.name][rand_pairs[:, 1]],
                self.origin_item_feature.name: input_feed[self.origin_item_feature.name][rand_pairs[:, 1]],
                self.pos.name: np.array(current_poss)[rand_pairs[:, 1]],
                self.keep_prob.name: 1.0})
            click_j_new = session.run(self.click,
                                      feed_dict={
                                          self.user_context_feature.name: input_feed[self.user_context_feature.name][rand_pairs[:, 1]],
                                          self.origin_item_feature.name: input_feed[self.origin_item_feature.name][rand_pairs[:, 1]],
                                          self.item_context_feature.name: input_feed[self.item_context_feature.name][
                                              rand_pairs[:, 1]],
                                          self.pos.name: np.array(current_poss)[rand_pairs[:, 0]],
                                          self.keep_prob.name: 1.0})
            click_j_log = origin_click_prob[rand_pairs[:, 1]]

            alpha = 0.7
            delta_revenue = ((click_i_new - click_i_old) / (click_i_log+1e-17) * click_i +  (click_j_new - click_j_old) / (click_j_log+1e-17) * click_j)
            pair_label = ((flatten_scores[rand_pairs[:, 0]] < flatten_scores[rand_pairs[:, 1]]).astype(float) - 0.5) * 2
            _, loss, summary = session.run([self.updates,  # Update Op that does SGD.
                                            self.loss,  # Loss for this batch.
                                            self.summary], feed_dict={
                self.pos_user_feature.name: input_feed[self.user_context_feature.name][rand_pairs[:, 0]],
                self.pos_item_feature.name: input_feed[self.origin_item_feature.name][rand_pairs[:, 0]],
                self.neg_user_feature.name: input_feed[self.user_context_feature.name][rand_pairs[:, 1]],
                self.neg_item_feature.name: input_feed[self.origin_item_feature.name][rand_pairs[:, 1]],
                self.deltaR.name: delta_revenue,
                self.pair_label.name: pair_label})



            return labels, scores, loss, summary, delta_revenue
        else:
            scores = []
            start = 0
            for sublabels in labels:
                seqlen = len(sublabels)
                scores.append(flatten_scores[start:start + seqlen].tolist())
                start = start + seqlen

            return labels, scores


    def KM_ranker(self, session, input_feed, labels, clicks, forward_only):
        scores = []
        start = 0
        for index, sublabels in enumerate(labels):
            seqlen = len(sublabels)
            click_table = np.reshape(session.run(self.click, feed_dict={
                self.user_context_feature.name: np.repeat(
                    input_feed[self.user_context_feature.name][start:start + seqlen, :], seqlen, axis=0),
                self.item_context_feature.name: np.repeat(
                    input_feed[self.item_context_feature.name][start:start + seqlen, :], seqlen,
                    axis=0),
                self.origin_item_feature.name: np.repeat(
                    input_feed[self.origin_item_feature.name][start:start + seqlen, :], seqlen, axis=0),
                self.pos.name: np.tile(np.arange(seqlen), seqlen),
                self.keep_prob.name: 1.0}), [seqlen, seqlen])

            perfect_match = KM(click_table, seqlen)
            scores.append((10 - perfect_match).tolist())
            start = start + seqlen
        return  labels, scores

    def KM_oracle(self, session, input_feed, labels, clicks, forward_only, oracle_model):
        scores = []
        start = 0
        for index, sublabels in enumerate(labels):
            seqlen = len(sublabels)

            click_table, _, _ = oracle_model.get_click_prob(np.repeat(input_feed[self.origin_item_feature.name][start:start + seqlen, :], seqlen, axis=0),
                                                       np.tile(np.arange(seqlen), seqlen),
                                                       np.repeat(np.array(sublabels), seqlen, axis=0))
            click_table = np.reshape(click_table, [seqlen, seqlen])
            perfect_match = KM(click_table, seqlen)
            scores.append((10 - perfect_match).tolist())
            start = start + seqlen
        return labels, scores, click_table



    def get_batch_for_bias(self, data_set):
        length = data_set.data.shape[0]
        rand_idx = np.random.randint(length, size=self.batch_size)
        data = data_set.data[rand_idx]  # (qid, did, pos, exam, click)
        user_ids = data[:, 0].astype(np.int)
        item_ids = data[:, 1].astype(np.int)
        user_features = data_set.query_features[user_ids]
        item_features = data_set.features[item_ids]
        positions = (data[:, 2] > 4).astype(np.int)

        # Create input feed map
        input_feed = {}
        input_feed[self.user_context_feature.name] = user_features
        input_feed[self.origin_item_feature.name] = item_features
        input_feed[self.pos.name] = positions

        return input_feed

    def get_batch_for_click(self, data_set, pos=-1):
        length = data_set.data.shape[0]
        rand_idx = np.random.randint(length, size=self.batch_size)
        data = data_set.data[rand_idx]  # (qid, did, pos, exam, click)
        user_ids = data[:, 0].astype(np.int)
        item_ids = data[:, 1].astype(np.int)
        user_context_features = data_set.query_features[user_ids]
        item_features = data_set.features[item_ids]
        item_context_features = data_set.doc_features[rand_idx]
        positions = data[:, 2] if pos == -1 else pos
        clicks = data[:, 4]
        # Create input feed map
        input_feed = {}
        input_feed[self.user_context_feature.name] = user_context_features
        input_feed[self.origin_item_feature.name] = item_features
        input_feed[self.item_context_feature.name] = item_context_features
        input_feed[self.pos.name] = positions
        input_feed[self.target_clicks.name] = clicks
        input_feed[self.pos_neg_ratio.name] = data_set.pos_neg_ratio

        return input_feed, data

    def get_batch_for_click_by_index(self, data_set, index, pos=-1):

        # rand_idx = index
        end_idx = min(data_set.data.shape[0], index+self.batch_size)
        rand_idx = np.array(range(index, end_idx)).astype(np.int)
        data = data_set.data[rand_idx]  # (qid, did, pos, exam, click label)
        user_ids = data[:, 0].astype(np.int)
        item_ids = data[:, 1].astype(np.int)
        user_context_features = data_set.query_features[user_ids]
        item_features = data_set.features[item_ids]
        item_context_features = data_set.doc_features[rand_idx]


        if pos!= -1:
            positions = pos
        else:
            positions = data[:, 2].astype(np.int)
        clicks = data[:, 4]

        # Create input feed map
        input_feed = {}
        input_feed[self.user_context_feature.name] = user_context_features
        input_feed[self.origin_item_feature.name] = item_features
        input_feed[self.item_context_feature.name] = item_context_features
        input_feed[self.pos.name] = np.array([positions]).reshape(-1)
        input_feed[self.target_clicks.name] = np.array([clicks]).reshape(-1)
        input_feed[self.pos_neg_ratio.name] = data_set.pos_neg_ratio

        return input_feed, data

    def prepare_data_with_index(self, data_set, index, user_features, item_features, item_context_features, labels, clicks):
        labels.append(data_set.relavance_labels[index])
        clicks.append(data_set.clicks[index])
        did_index = sum(data_set.len_list[: index])
        for i in range(len(labels[-1])):
            user_features.append(data_set.query_features[data_set.qids[index]])
            item_features.append(data_set.features[data_set.dids[index][i]])
            item_context_features.append(data_set.doc_features[did_index + i])


    def get_batch_for_ranker(self, data_set):
        # print('Begin data loading...')
        length = len(data_set.qids)


        user_features, item_features, item_context_features, labels,clicks = [], [], [], [],[]
        if not self.data_permutation:
            self.data_permutation = [i for i in range(length)]
        rank_list_idxs = []


        for _ in range(self.batch_size):
            if self.data_step >= length:
                random.shuffle(self.data_permutation)
                self.data_step = self.data_step % length
            rank_list_idxs.append(self.data_permutation[self.data_step])
            self.prepare_data_with_index(data_set, self.data_step, user_features, item_features, item_context_features, labels,clicks)
            self.data_step += 1

        # Create input feed map
        input_feed = {}
        input_feed[self.user_context_feature.name] = np.array(user_features)
        input_feed[self.item_context_feature.name] = np.array(item_context_features)
        input_feed[self.origin_item_feature.name] = np.array(item_features)

        return input_feed, labels, clicks

    def get_batch_for_ranker_by_index(self, data_set, i):
        # print('Begin data loading...')
        length = len(data_set.qids)
        user_features, item_features, item_context_features, labels, clicks = [], [], [], [],[]

        self.prepare_data_with_index(data_set, i, user_features, item_features, item_context_features, labels, clicks)


        # Create input feed map
        input_feed = {}
        input_feed[self.user_context_feature.name] = np.array(user_features)
        input_feed[self.item_context_feature.name] = np.array(item_context_features)
        input_feed[self.origin_item_feature.name] = np.array(item_features)


        return input_feed, labels, clicks



if __name__ == "__main__":
   pass

