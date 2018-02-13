# -*- coding:utf-8 -*-
import numpy as np
import json
import os
import sys
import argparse
import pickle
from sklearn.svm import SVC


def main(args):
    labels_file = open(os.path.join(args.data_dir,'test_labels.txt'),'r')
    embs_file = open(os.path.join(args.data_dir,'test_embs.txt'),'r')
     
    labels = json.loads(labels_file.read())
    #emb_array = json.loads(embs_file.read())

    #labels = np.array(json.loads(labels_file.read()))
    emb_array = np.array(json.loads(embs_file.read()))
    emb_array.reshape(-1,128)
     
    print(len(labels))
    print(len(emb_array),emb_array.shape)

    classifier_filename_exp = os.path.join(args.models_out_dir,'star_classifier.pkl')

    # Classify images
    print('Testing classifier')
    with open(classifier_filename_exp, 'rb') as infile:
        #(model, class_names) = pickle.load(infile)
        model = pickle.load(infile)

        print('Loaded classifier model from file "%s"' % classifier_filename_exp)

        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
         
        #for i in range(len(best_class_indices)):
        #    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                 
        accuracy = np.mean(np.equal(best_class_indices, labels))
        print('Accuracy: %.3f' % accuracy)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
     
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing embs and labels.', default='/mnt/sdb1/datasets/star/repr_160')
    parser.add_argument('--models_out_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='/mnt/sdb1/datasets/star/face_classifier')
     
    return parser.parse_args(argv)
     
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

"""
16461
16461
Testing classifier
Loaded classifier model from file "/mnt/sdb1/datasets/star/face_classifier/star_classifier.pkl"
Accuracy: 0.891
"""
