# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import json
import os
import sys
import argparse
import math

def onehot(labels):
    ''' one-hot 编码 '''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
     
    return onehot_labels

def load_data(args):
    labels_file = open(os.path.join(args.data_dir,'test_labels.txt'),'r')
    embs_file = open(os.path.join(args.data_dir,'test_embs.txt'),'r')
     
    labels = json.loads(labels_file.read())
    embs = json.loads(embs_file.read())
     
    print(len(labels))
    print(len(embs))
     
    return  embs,labels

def multiplayer_perceptron(x, weight, bias):
    # _,n_input    -> _,n_hidden_1
    layer1 = tf.add(tf.matmul(x, weight['h1']), bias['h1'])
    layer1 = tf.nn.relu(layer1) 
    # _,n_hidden_1 -> _,n_hidden_2
    layer2 = tf.add(tf.matmul(layer1, weight['h2']), bias['h2'])
    layer2 = tf.nn.relu(layer2)
    # _,n_hidden_2 -> _,n_hidden_3
    layer3 = tf.add(tf.matmul(layer2, weight['h3']), bias['h3'])
    layer3 = tf.nn.relu(layer3)
    # _,n_hidden_3 -> _,n_class
    out_layer = tf.add(tf.matmul(layer3, weight['out']), bias['out'],name='pred_out')

    return out_layer

def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
         
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
         
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            load_model(args.model_path)

            embs,labels = load_data(args)
            predict(args,sess,embs,labels)

def predict(args,sess,embs,labels):

    X_train = np.array(embs)
    n_sample,n_input = X_train.shape
    print("input dim:{}*{}".format(n_sample,n_input))

    y_train = onehot(labels)
    n_sample_y,n_class = y_train.shape
    print("output dim:{}*{}".format(n_sample_y,n_class))

    # Get input and output tensors
    embs_placeholder = tf.get_default_graph().get_tensor_by_name("embs_in:0")
    cls_placeholder = tf.get_default_graph().get_tensor_by_name("cls_out:0")
    n_class = cls_placeholder.get_shape()[1]
    print("cls shape:{}".format(cls_placeholder.get_shape()))
    pred = tf.get_default_graph().get_tensor_by_name("pred_out:0")
    print("pred shape:{}".format(pred.get_shape()))

    nrof_batch = int(math.ceil(1.0*n_sample / args.batch_size))

    pred_array = np.zeros((n_sample,n_class))

    for i in range(nrof_batch):
        start_index = i*args.batch_size
        end_index = min((i+1)*args.batch_size, n_sample)
        embs_batch = X_train[start_index:end_index,:]
        pred_array[start_index:end_index,:] = sess.run(pred,feed_dict={embs_placeholder:embs_batch})

    correct_prediction = np.equal(np.argmax(pred_array, 1), np.argmax(y_train, 1))
    accuracy = np.mean(correct_prediction.astype(float))
    print("accuracy:{}".format(accuracy))


def train(args,sess,embs,labels):

    X_train = np.array(embs)
    n_sample,n_input = X_train.shape
    print("input dim:{}*{}".format(n_sample,n_input))

    y_train = onehot(labels)
    n_sample_y,n_class = y_train.shape
    print("output dim:{}*{}".format(n_sample_y,n_class))

    x = tf.placeholder('float', [None, n_input],name='embs_in')
    y = tf.placeholder('float', [None, n_class],name='cls_out')
     
    n_hidden_1 = 1024
    n_hidden_2 = 1024
    n_hidden_3 = 1024
     
    weight = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_hidden_3, n_class]))
    }
    bias = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_class]))
    }

    # 建立模型
    pred = multiplayer_perceptron(x, weight, bias)

    # 定义损失函数
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))

    # 优化
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
     
    # 初始化所有变量
    init = tf.initialize_all_variables()
     
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
     
    # Save model
    saver = tf.train.Saver(weight.values()+bias.values())

    # 训练模型
    training_epochs = 1000#500
    batch_size = 500
    display_step = 1

    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_sample / batch_size)

        for i in range(total_batch):
            _, c = sess.run([optimizer, cost], feed_dict={x: X_train[i*batch_size : (i+1)*batch_size, :],
                                                          y: y_train[i*batch_size : (i+1)*batch_size, :]})
            avg_cost += c / total_batch

        #plt.plot(epoch+1, avg_cost, 'co')

        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost=', '{:.9f}'.format(avg_cost))

    saver.save(sess,os.path.join(args.models_out_dir,"star_classifier"),global_step=training_epochs)
    print('Opitimization Finished!')
     
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
     
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing embs and labels.', default='/mnt/sdb1/datasets/star/repr_160')
    parser.add_argument('--model_path', type=str,
        help='Directory contains trained models and checkpoints.', default='/mnt/sdb1/datasets/star/face_classifier')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=500)
     
    return parser.parse_args(argv)
     
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

