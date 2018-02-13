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

    x = tf.placeholder('float', [None, n_input],name='embs_in')
    y = tf.placeholder('float', [None, n_class],name='cls_out')
     
    keep_prob = tf.placeholder(tf.float32) #Dropout失活率

    n_hidden_1 = 128
    #n_hidden_1 = 768
    #n_hidden_1 = 896
    #n_hidden_1 = 1024
    #n_hidden_1 = 1152
    n_hidden_2 = 512
    n_hidden_3 = 1024

    weight = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_class]))
        #'out': tf.Variable(tf.random_normal([n_input, n_class]))
    }
    bias = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_class]))
    }
    """
    weight = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden_1, n_class],stddev=0.1))
    }
    bias = {
        'h1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_class],stddev=0.1))
    }
    """
    # 建立模型
    pred = multiplayer_perceptron(x, weight, bias,keep_prob)

    # 定义损失函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

    # 优化
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
     
    #optimizer = tf.train.AdagradOptimizer(0.3).minimize(cost)  
    
    #optimizer = tf.train.RMSPropOptimizer(0.001).minimize(cost)

    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

    # 初始化所有变量
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
     
    # Save model
    saver = tf.train.Saver(weight.values()+bias.values())

    # 训练模型
    training_epochs = args.max_nrof_epochs
    batch_size = args.batch_size
    display_step = 1

    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_sample / batch_size)

        for i in range(total_batch):
            _, c,a = sess.run([optimizer, cost,accuracy], feed_dict={x: X_train[i*batch_size : (i+1)*batch_size, :],
                                                          y: y_train[i*batch_size : (i+1)*batch_size, :],
                                                          keep_prob:1.0})
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

