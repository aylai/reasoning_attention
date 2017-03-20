# coding: utf-8

# In[1]:

from __future__ import print_function

import pickle
import sys, os

import numpy
import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import lasagne
import time

import argparse

from custom_layers import CustomEmbedding, CustomLSTMEncoder, CustomDense, CustomLSTMDecoder


# In[2]:

def prepare(df):
    seqs_premise = []
    seqs_hypothesis = []
    for cc in df['sentence1']:
        seqs_premise.append(cc)
    for cc in df['sentence2']:
        seqs_hypothesis.append(cc)
    seqs_p = seqs_premise
    seqs_h = seqs_hypothesis

    lengths_p = [len(s) for s in seqs_p]
    lengths_h = [len(s) for s in seqs_h]

    n_samples = len(seqs_p)
    maxlen_p = numpy.max(lengths_p) + 1
    maxlen_h = numpy.max(lengths_h) + 1

    premise = numpy.zeros((n_samples, maxlen_p))
    hypothesis = numpy.zeros((n_samples, maxlen_h))
    premise_masks = numpy.zeros((n_samples, maxlen_p))
    hypothesis_masks = numpy.zeros((n_samples, maxlen_h))
    for idx, [s_t, s_h] in enumerate(zip(seqs_p, seqs_h)):
        assert lengths_h[idx] == len(s_h)
        premise[idx, :lengths_p[idx]] = s_t
        premise_masks[idx, :lengths_p[idx]] = 1
        hypothesis[idx, :lengths_h[idx]] = s_h
        hypothesis_masks[idx, :lengths_h[idx]] = 1
    labels = []
    for gl in df['gold_label']:
        if gl == 'entailment':
            value = 2
        elif gl == 'contradiction':
            value = 1
        elif gl == 'neutral':
            value = 0
        else:
            raise ValueError('unknown gold_label {0}'.format(gl))
        labels.append(value)

    labels = np.array(labels)
    return (premise.astype('int32'),
            premise_masks.astype('int32'),
            hypothesis.astype('int32'),
            hypothesis_masks.astype('int32'),
            labels.astype('int32'))


# In[3]:

def load_data(params):
    print('Loading data ...')
    train_df, dev_df, test_df = (None, None, None)
    # with open('./snli/converted_train.pkl', 'rb') as f:
    splits = {'train': 0, 'test': 0, 'dev': 0}
    splits[params['train_split']] += 1
    splits[params['test_split']] += 1
    splits[params['dev_split']] += 1
    if splits['train'] > 0:
        with open(params['data_dir'] + '/converted_train.pkl', 'rb') as f:
            print('Loading train ...')
            train_df = pickle.load(f)
            if params['data_dir'] != 'snli':
                train_df = pd.DataFrame(data=train_df)
            print(len(train_df))
            filtered_s2 = train_df.sentence2.apply(lambda s2: len(s2) != 0)
            train_df = train_df[filtered_s2]
            print(len(train_df))
            train_df = train_df[train_df.gold_label != '-']
            train_df = train_df.reset_index()
            print(len(train_df))
            # prepare(train_df)
    if splits['dev'] > 0:
        with open(params['data_dir'] + '/converted_dev.pkl', 'rb') as f:
            print('Loading dev ...')
            dev_df = pickle.load(f)
            if params['data_dir'] != 'snli':
                dev_df = pd.DataFrame(data=dev_df)
            print(len(dev_df))
            filtered_s2 = dev_df.sentence2.apply(lambda s2: len(s2) != 0)
            dev_df = dev_df[filtered_s2]
            print(len(dev_df))
            dev_df = dev_df[dev_df.gold_label != '-']
            dev_df = dev_df.reset_index()
            print(len(dev_df))
    if splits['test'] > 0:
        with open(params['data_dir'] + '/converted_test.pkl', 'rb') as f:
            print('Loading test ...')
            test_df = pickle.load(f)
            if params['data_dir'] != 'snli':
                test_df = pd.DataFrame(data=test_df)
            print(len(test_df))
            filtered_s2 = test_df.sentence2.apply(lambda s2: len(s2) != 0)
            test_df = test_df[filtered_s2]
            print(len(test_df))
            test_df = test_df[test_df.gold_label != '-']
            test_df = test_df.reset_index()
            print(len(test_df))
    return train_df, dev_df, test_df


def precision_recall(predicted, labels):
    # if self.data_type == "snli" or self.data_type == "mpe_concat" or self.data_type == "mpe_indiv":
    correct = 0
    total_pred = 0
    true_pos = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    false_pos = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    false_neg = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    label_map = {0: "neutral", 1: "contradiction", 2: "entailment"}
    for idx, pred_label_id in enumerate(predicted):
        total_pred += 1
        true_label_id = labels[idx]
        if pred_label_id == true_label_id:
            true_pos[label_map[pred_label_id]] += 1
        else:
            false_pos[label_map[pred_label_id]] += 1
            false_neg[label_map[true_label_id]] += 1
    precision = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    recall = {'entailment': 0, 'neutral': 0, 'contradiction': 0}
    for label in ['entailment', 'neutral', 'contradiction']:
        # print(label, ":", str(true_pos[label] + false_neg[label]))
        prec_sum = 1.0 * (true_pos[label] + false_pos[label])
        if prec_sum > 0:
            precision[label] = true_pos[label] / prec_sum
        else:
            precision[label] = 0.0
        recall_sum = 1.0 * (true_pos[label] + false_neg[label])
        if recall_sum > 0:
            recall[label] = true_pos[label] / recall_sum
        else:
            recall[label] = 0.0
    result = {"precision": precision, "recall": recall}
    return result
# In[7]:

premise_max = 82 + 1
hypothesis_max = 62 + 1


# In[8]:

# def main(num_epochs=10, k=100,
#          batch_size=128,
#          display_freq=100,
#          save_freq=1000,
#          load_previous=False,
#          load_epoch=0,
#          attention=True,
#          word_by_word=True,
#          mode='word_by_word',
#          stage='train',
#          ):
def main(params, load_model=None):
    num_epochs = params['num_epochs']
    print('num_epochs: {}'.format(num_epochs))
    k = params['lstm_dim']
    print('k: {}'.format(k))
    batch_size = params['batch_size']
    print('batch_size: {}'.format(batch_size))
    display_freq = params['display_freq']
    print('display_frequency: {}'.format(display_freq))
    save_freq = 10000
    print('save_frequency: {}'.format(save_freq))
    modeldir = os.path.join(params["run_dir"], params["exp_name"])
    save_filename = os.path.join(modeldir, "save")
    # print('load previous: {}'.format(load_previous))
    # print('start epoch: {}'.format(load_epoch))
    attention = params['attention']
    print('attention: {}'.format(attention))
    word_by_word = params['word_by_word']
    print('word_by_word: {}'.format(word_by_word))
    mode = params['model_type']
    # save_filename = './snli/{}_model.npz'.format(params['model'])
    train_df, dev_df, test_df = load_data(params)
    print("Building network ...")
    premise_var = T.imatrix('premise_var')
    premise_mask = T.imatrix('premise_mask')
    hypo_var = T.imatrix('hypo_var')
    hypo_mask = T.imatrix('hypo_mask')
    unchanged_W = pickle.load(open(params['data_dir'] + '/unchanged_W.pkl', 'rb'))
    unchanged_W = unchanged_W.astype('float32')
    unchanged_W_shape = unchanged_W.shape
    oov_in_train_W = pickle.load(open(params['data_dir'] + '/oov_in_train_W.pkl', 'rb'))
    oov_in_train_W = oov_in_train_W.astype('float32')
    oov_in_train_W_shape = oov_in_train_W.shape
    print('unchanged_W.shape: {0}'.format(unchanged_W_shape))
    print('oov_in_train_W.shape: {0}'.format(oov_in_train_W_shape))
    # best hypoparameters
    p = 0.2
    learning_rate = 0.001
    # learning_rate = 0.0003
    # l2_weight = 0.0003
    l2_weight = 0.

    l_premise = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_var)
    l_premise_mask = lasagne.layers.InputLayer(shape=(None, premise_max), input_var=premise_mask)
    l_hypo = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_var)
    l_hypo_mask = lasagne.layers.InputLayer(shape=(None, hypothesis_max), input_var=hypo_mask)

    premise_embedding = CustomEmbedding(l_premise, unchanged_W, unchanged_W_shape,
                                        oov_in_train_W, oov_in_train_W_shape,
                                        p=p)
    # weights shared with premise_embedding
    hypo_embedding = CustomEmbedding(l_hypo, unchanged_W=premise_embedding.unchanged_W,
                                     unchanged_W_shape=unchanged_W_shape,
                                     oov_in_train_W=premise_embedding.oov_in_train_W,
                                     oov_in_train_W_shape=oov_in_train_W_shape,
                                     p=p,
                                     dropout_mask=premise_embedding.dropout_mask)

    l_premise_linear = CustomDense(premise_embedding, k,
                                   nonlinearity=lasagne.nonlinearities.linear)
    l_hypo_linear = CustomDense(hypo_embedding, k,
                                W=l_premise_linear.W, b=l_premise_linear.b,
                                nonlinearity=lasagne.nonlinearities.linear)

    encoder = CustomLSTMEncoder(l_premise_linear, k, peepholes=False, mask_input=l_premise_mask)
    decoder = CustomLSTMDecoder(l_hypo_linear, k, cell_init=encoder, peepholes=False, mask_input=l_hypo_mask,
                                encoder_mask_input=l_premise_mask,
                                attention=attention,
                                word_by_word=word_by_word
                                )
    if p > 0.:
        print('apply dropout rate {} to decoder'.format(p))
        decoder = lasagne.layers.DropoutLayer(decoder, p)
    l_softmax = lasagne.layers.DenseLayer(
            decoder, num_units=3,
            nonlinearity=lasagne.nonlinearities.softmax)
    if load_model is not None:
        load_filename = os.path.join(modeldir, load_model + '.npz')
    # if load_previous:
        print('loading previous saved model ...')
        # And load them again later on like this:
        # load_filename = './snli/{}_model_epoch{}.npz'.format(mode, load_epoch)
        with np.load(load_filename) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(l_softmax, param_values)

    target_var = T.ivector('target_var')

    # lasagne.layers.get_output produces a variable for the output of the net
    prediction = lasagne.layers.get_output(l_softmax, deterministic=False)
    # The network output will have shape (n_batch, 3);
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    cost = loss.mean()
    if l2_weight > 0.:
        # apply l2 regularization
        print('apply l2 penalty to all layers, weight: {}'.format(l2_weight))
        regularized_layers = {encoder: l2_weight,
                              decoder: l2_weight}
        l2_penalty = lasagne.regularization.regularize_network_params(l_softmax,
                                                                      lasagne.regularization.l2) * l2_weight
        cost += l2_penalty
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_softmax, trainable=True)
    # Compute adam updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adam(cost, all_params, learning_rate=learning_rate)

    test_prediction = lasagne.layers.get_output(l_softmax, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # lasagne.objectives.categorical_accuracy()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train_fn = theano.function([premise_var, premise_mask, hypo_var, hypo_mask, target_var],
                               cost, updates=updates)
    val_fn = theano.function([premise_var, premise_mask, hypo_var, hypo_mask, target_var],
                             [test_loss, test_acc, test_prediction, target_var])
    split_data = {'train': train_df, 'test': test_df, 'dev': dev_df}
    train_data = split_data[params['train_split']]
    test_data = split_data[params['test_split']]
    dev_data = split_data[params['dev_split']]
    if params['stage'] == 'train':
        print("Training ...")

        print('train_df.shape: {0}'.format(train_data.shape))
        print('dev_df.shape: {0}'.format(dev_data.shape))
        print('test_df.shape: {0}'.format(test_data.shape))
        try:
            # Finally, launch the training loop.
            print("Starting training...")
            # We iterate over epochs:
            for epoch in range(num_epochs):
                # In each epoch, we do a full pass over the training data:
                shuffled_train_df = train_data.reindex(np.random.permutation(train_data.index))
                train_err = 0
                train_acc = 0
                train_batches = 0
                start_time = time.time()
                display_at = time.time()
                save_at = time.time()
                for start_i in range(0, len(shuffled_train_df), batch_size):
                    batched_df = shuffled_train_df[start_i:start_i + batch_size]
                    ps, p_masks, hs, h_masks, labels = prepare(batched_df)
                    train_err += train_fn(ps, p_masks, hs, h_masks, labels)
                    err, acc, _, _ = val_fn(ps, p_masks, hs, h_masks, labels)
                    train_acc += acc
                    train_batches += 1
                    # display
                    if train_batches % display_freq == 0:
                        print("Seen {:d} samples, time used: {:.3f}s".format(
                            start_i + batch_size, time.time() - display_at))
                        print("  current training loss:\t\t{:.6f}".format(train_err / train_batches))
                        print("  current training accuracy:\t\t{:.6f}".format(train_acc / train_batches))
                    # do tmp save model
                    if train_batches % save_freq == 0:
                        print('saving to {}, time used {:.3f}s'.format(save_filename, time.time() - save_at))
                        np.savez(save_filename + '.npz',
                                 *lasagne.layers.get_all_param_values(l_softmax))
                        save_at = time.time()

                # And a full pass over the validation data:
                val_err = 0
                val_acc = 0
                val_batches = 0
                predictions = []
                targets = []
                for start_i in range(0, len(dev_data), batch_size):
                    batched_df = dev_data[start_i:start_i + batch_size]
                    ps, p_masks, hs, h_masks, labels = prepare(batched_df)
                    err, acc, pred, target = val_fn(ps, p_masks, hs, h_masks, labels)
                    predictions.extend(T.argmax(pred, axis=1).eval().tolist())
                    targets.extend(target.tolist())
                    val_err += err
                    val_acc += acc
                    val_batches += 1

                # Then we print the results for this epoch:
                print("Epoch {} of {} took {:.3f}s".format(
                        epoch + 1, num_epochs, time.time() - start_time))
                print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
                print("  training accuracy:\t\t{:.2f} %".format(
                        train_acc / train_batches * 100))
                print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
                print("  validation accuracy:\t\t{:.2f} %".format(
                        val_acc / val_batches * 100))
                print(precision_recall(predictions, targets))
                temp_save_filename = save_filename + '_' + str(epoch + 1) + '.npz'
                print('saving to {}'.format(temp_save_filename))
                np.savez(temp_save_filename,
                         *lasagne.layers.get_all_param_values(l_softmax))

            # Optionally, you could now dump the network weights to a file like this:
            # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
            #
            # And load them again later on like this:
            # with np.load('model.npz') as f:
            #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            # lasagne.layers.set_all_param_values(network, param_values)
        except KeyboardInterrupt:
            print('exit ...')
    else:
        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        predictions = []
        targets = []
        for start_i in range(0, len(test_df), batch_size):
            batched_df = test_df[start_i:start_i + batch_size]
            ps, p_masks, hs, h_masks, labels = prepare(batched_df)
            err, acc, pred, target = val_fn(ps, p_masks, hs, h_masks, labels)
            predictions.extend(T.argmax(pred, axis=1).eval().tolist())
            targets.extend(target.tolist())
            # print(.eval())
            # print(target)
            # print(T.eq(T.argmax(pred, axis=1), target).eval())
            # print("***")
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
        print(precision_recall(predictions, targets))


# In[9]:

if __name__ == '__main__':
    start = time.time()

    dirname, filename = os.path.split(os.path.abspath(__file__))
    # GIT_DIR = "/".join(dirname.split("/")[:-1])
    RUNS_DIR = os.path.join(dirname, "runs")
    DATA_DIR = os.path.join(dirname, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="active this flag to train the model")
    parser.add_argument("--test", action="store_true", help="active this flag to test the model")
    parser.add_argument("--train_split", type=str, default="train", help="data split to train model")
    parser.add_argument("--dev_split", type=str, default="dev", help="data split for dev evaluation")
    parser.add_argument("--test_split", type=str, default="test", help="data split to evaluate")
    parser.add_argument("--saved_model", help="name of saved model")
    parser.add_argument("--data_dir", default="snli", help="path to the SNLI dataset directory")
    # parser.add_argument("--vector_type", default="glove", help="specify vector type glove/word2vec/none")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_type", type=str, default="attention")

    # parser.add_argument("--data_type", type=str, default="snli", help="snli or mpe")

    # parser.add_argument("--learning_rate", type=float, default=0.001)
    # parser.add_argument("--dropout", type=float, default=0.8)
    # parser.add_argument("--l2_reg", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    # parser.add_argument("--embedding_dim", type=int, default=300, help="vector dimension")
    parser.add_argument("--lstm_dim", type=int, default=100, help="LSTM output dimension (k in the original paper)")
    args = parser.parse_args()

    if args.model_name is None:
        print("Specify name of experiment")
        sys.exit(0)

    params = {
        'run_dir': RUNS_DIR,
        'exp_name': args.model_name,
        'load_model_file': args.saved_model,
        'data_dir': os.path.join(DATA_DIR, args.data_dir),

        'train_split': args.train_split,
        'dev_split': args.dev_split,
        'test_split': args.test_split,

        'model_type': args.model_type,
        # 'glove_file': args.glove_path,
        # 'w2v_file': args.word2vec_path,
        # 'vector_src': args.vector_type,

        # 'data_type': args.data_type,
        # 'premise_selection': args.premise_selection,

        # 'embedding_dim': args.embedding_dim,  # word embedding dim
        # 'oov_op': args.oov_operation,  # save, load, none
        # 'oov_file': args.model_name + '.vector',

        'batch_size': args.batch_size,
        'lstm_dim': args.lstm_dim,
        # 'dropout': args.dropout,  # 1 = no dropout, 0.5 = dropout
        # 'multicell': args.multicell,

        # 'learning_rate': args.learning_rate,
        # 'l2_reg': args.l2_reg,

        'num_epochs': args.num_epochs,
        # 'num_classes': args.num_classes,
    }

    if args.train:
        params['stage'] = 'train'
    elif args.test:
        params['stage'] = 'test'
    else:
        print("Choose to train or test model.")
        sys.exit(0)

    # method
    params['attention'] = True
    params['word_by_word'] = True
    if params['model_type'] == 'condition':
        params['attention'] = False
        params['word_by_word'] = False
    elif params['model_type'] == 'attention':
        params['word_by_word'] = False
    elif params['model_type'] == 'word_by_word':
        params['attention'] = True
        params['word_by_word'] = True
    else:
        print('doesn\'t recognize mode {}'.format(params['model_type']))
        print('only supports [condition|attention|word_by_word]')
        sys.exit(1)

    if not os.path.exists(params['run_dir']):
        os.mkdir(params['run_dir'])
    modeldir = os.path.join(params['run_dir'], params["exp_name"])
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    # load=False
    # if len(sys.argv) > 2 and sys.argv[2] == "load":
    #     load = True
    # start_epoch=0
    # if len(sys.argv) > 3:
    #     start_epoch = int(sys.argv[3])
    params['display_freq'] = 1000

    if params['stage'] == 'train':
        main(params)
    elif params['stage'] == 'test':
        main(params, load_model=params['load_model_file'])
        # num_epochs=args.num_epochs, batch_size=args.batch_size,
        #  display_freq=1000,
        #  attention=attention,
        #  word_by_word=word_by_word,
        #  mode=mode,
        #  stage=stage,
        #  load_filename=args.saved_model)

    end = time.time() - start
    m, s = divmod(end, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))
