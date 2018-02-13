# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import os
import sys
import argparse

def onehot(labels):
    ''' one-hot 编码 '''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
     
    return onehot_labels

def load_data(args):
    labels_file = open(os.path.join(args.data_dir,'train_labels.txt'),'r')
    embs_file = open(os.path.join(args.data_dir,'train_embs.txt'),'r')
     
    labels = json.loads(labels_file.read())
    embs = json.loads(embs_file.read())
     
    print(len(labels))
    print(len(embs))
     
    return  embs,labels

def multiplayer_perceptron(x, weight, bias,keep_prob):
    # _,n_input    -> _,n_hidden_1
    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
    layer1 = tf.nn.relu(layer1) 
    layer1_drop = tf.nn.dropout(layer1,keep_prob)
    # _,n_hidden_1 -> _,n_hidden_2
    #layer2 = tf.add(tf.matmul(layer1, weight['h2']), bias['h2'])
    #layer2 = tf.nn.relu(layer2)
    #print("layer2 shape:{}".format(layer2.get_shape()))
    # _,n_hidden_2 -> _,n_hidden_3
    #layer3 = tf.add(tf.matmul(layer2, weight['h3']), bias['h3'])
    #layer3 = tf.nn.relu(layer3)
    # _,n_hidden_3 -> _,n_class
    
    out_layer = tf.add(tf.matmul(layer1_drop, weight['out']), bias['out'],name='pred_out')
    #out_layer = tf.add(tf.matmul(x, weight['out']), bias['out'],name='pred_out')

    return out_layer

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            embs,labels = load_data(args)
            train(args,sess,embs,labels)

def train(args,sess,embs,labels):

    X_train = np.array(embs)
    n_sample,n_input = X_train.shape
    print("input dim:{}*{}".format(n_sample,n_input))

    y_train = onehot(labels)
    n_sample_y,n_class = y_train.shape
    print("output dim:{}*{}".format(n_sample_y,n_class))

    #x = tf.placeholder('float', [None, n_input],name='embs_in')
    #y = tf.placeholder('float', [None, n_class],name='cls_out')
     
    with tf.name_scope('input'):
        # [BATCH_SIZE, NUM_FEATURES]
        x_input = tf.placeholder(dtype=tf.float32, shape=[None, n_input], name='x_input')

        # [BATCH_SIZE]
        y_input = tf.placeholder(dtype=tf.uint8, shape=[None], name='y_input')

        # [BATCH_SIZE, NUM_CLASSES]
        y_onehot = tf.one_hot(indices=y_input, depth=n_class, on_value=1, off_value=-1,
                              name='y_onehot')

    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

    with tf.name_scope('training_ops'):
        with tf.name_scope('weights'):
            weight = tf.get_variable(name='weights',
                     initializer=tf.random_normal([n_input, n_class],stddev=0.01))
            variable_summaries(weight)
        with tf.name_scope('biases'):
            bias = tf.get_variable(name='biases', initializer=tf.constant([0.1], shape=[n_class]))
            variable_summaries(bias)
        with tf.name_scope('Wx_plus_b'):
            output = tf.matmul(x_input, weight) + bias
            tf.summary.histogram('pre-activations', output)

        with tf.name_scope('svm'):
            regularization = tf.reduce_mean(tf.square(weight))
            hinge_loss = tf.reduce_mean(tf.square(tf.maximum(tf.zeros([args.batch_size, n_class]),
                                                                 1 - tf.cast(y_onehot, tf.float32) * output)))
            with tf.name_scope('loss'):
                loss = regularization + args.svm_c * hinge_loss
        tf.summary.scalar('loss', loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.name_scope('accuracy'):
            predicted_class = tf.sign(output)
            predicted_class = tf.identity(predicted_class, name='prediction')
            with tf.name_scope('correct_prediction'):
                correct = tf.equal(tf.argmax(predicted_class, 1), tf.argmax(y_onehot, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

    # initialize the variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
     
    # Save model
    saver = tf.train.Saver(weight.values()+bias.values())

    # 训练模型
    training_epochs = args.max_nrof_epochs
    batch_size = args.batch_size
    display_step = 1

    sess.run(init_op)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_sample / batch_size)

        for i in range(total_batch):
            feed_dict = {x_input: X_train[i*batch_size : (i+1)*batch_size, :],
                         y_input: labels[i*batch_size : (i+1)*batch_size],
                         learning_rate:0.001}
            _, c,a = sess.run([optimizer, loss,accuracy], feed_dict)
            avg_cost += c / total_batch

        #plt.plot(epoch+1, avg_cost, 'co')

        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost),'accuracy:{:0.4f}'.format(a))

    saver.save(sess,os.path.join(args.models_out_dir,"star_classifier"),global_step=training_epochs)
    print('Opitimization Finished!')
     
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
     
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing embs and labels.', default='/mnt/sdb1/datasets/star/repr_160')
    parser.add_argument('--models_out_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='/mnt/sdb1/datasets/star/face_classifier')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
     
    return parser.parse_args(argv)
     
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

