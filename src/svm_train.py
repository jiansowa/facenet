# -*- coding:utf-8 -*-
import numpy as np
import json
import os
import sys
import argparse
import pickle
from sklearn.svm import SVC


def main(args):
    labels_file = open(os.path.join(args.data_dir,'train_labels.txt'),'r')
    embs_file = open(os.path.join(args.data_dir,'train_embs.txt'),'r')
     
    labels = json.loads(labels_file.read())
    #emb_array = json.loads(embs_file.read())

    #labels = np.array(json.loads(labels_file.read()))
    emb_array = np.array(json.loads(embs_file.read()))
    emb_array.reshape(-1,128)
     
    print(len(labels))
    print(len(emb_array),emb_array.shape)

    # Train classifier
    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(emb_array, labels)
            
    # Create a list of class names
    #class_names = [ cls.name.replace('_', ' ') for cls in dataset]

    # Saving classifier model
    classifier_filename_exp = os.path.join(args.models_out_dir,'star_classifier.pkl')
    with open(classifier_filename_exp, 'wb') as outfile:
        #pickle.dump((model, class_names), outfile)
        pickle.dump(model, outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
     
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing embs and labels.', default='/mnt/sdb1/datasets/star/repr_160')
    parser.add_argument('--models_out_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='/mnt/sdb1/datasets/star/face_classifier')
     
    return parser.parse_args(argv)
     
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

