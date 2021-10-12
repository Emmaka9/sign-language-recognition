from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import os
import pickle
import sys
import tensorflow as tf
from tqdm import tqdm


def load_graph(model_file):
    '''
    1. take a graph obj. graph = tf.Graph()
    2. take graph def obj to define the graph graph_def = tf.GraphDef()
    3. open the model file in read mode, and read(parse) it into graph def as that defines the graph
    4. import the definition into the graph
    5. return the graph
    '''
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    # 3.
    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    '''
    4.
    creates a new graph and places everything (declared inside its scope) into this graph. If
    the graph is the only graph, it's useless. But it's a good practice because if you start to
    work with many graphs it's easier to understand where ops and vars are placed. Since this
    statement costs you nothing, it's better to write it anyway. Just to be sure that if you
    refactor the code in the future, the operations defined belong to the graph you choose
    initially.
    '''
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def get_tensor_from_image(frames: list, input_height=299, input_width=299, input_mean=0, input_std=255):
    input_name = "file_reader"

    '''
    func parameter frames: batch: ['training_frames/away/010.jpeg', 'training_frames/away/011.jpeg', ...] 
    
    tf.read_file(filename, name=None): returns a tensor with the entire contents ofthe input filename.
    doesn't do any parsing, just returns the contents as they are.
    
    frames: [(a tensor('training_frames/away/010.jpeg', opName), 'training_frames/away/010.jpeg'), (tensor, 'training_frames/away/011.jpeg') ...]
    '''
    frames = [(tf.read_file(filename=frame, name=input_name), frame) for frame in frames]
    decoded_frames = []
    for frame in frames:
        file_name = frame[1]
        file_reader = frame[0]
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
        decoded_frames.append(image_reader)
    float_caster = [tf.cast(image_reader, tf.float32) for image_reader in decoded_frames]
    float_caster = tf.stack(float_caster)
    resized = tf.image.resize_bilinear(float_caster, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    with tf.Session() as sess:
        result = sess.run(normalized)
    # result: a numpy n-dim array
    return result


def load_labels(label_file):
    ''' returns label: a list containing the labels. '''
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def predict(graph, image_tensor, input_layer, output_layer):
    ''' takes an image tensor, evalueates it.
    '''

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer

    # returns operation with the given name.
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    with tf.Session(graph=graph) as sess:
        results = sess.run(
            output_operation.outputs[0],
            {input_operation.outputs[0]: image_tensor} #dictionary
        )
    results = np.squeeze(results)
    return results


def predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255

    graph = load_graph(model_file)
    
    # list of the labels. *each label is a dir with frames inside.
    labels_in_dir = os.listdir(frames_folder)
    
    '''
    frames:- a list of tuples: each tuple has 3 elements: dirpath, dirnames, fileNames.
    In frames, only those dirs exist if the basename of those dirpath exist in labels_in_dirs.
    
    frames = [('/home/emmaka/Documents/Programming/comp-asl/final/SLR/train_frames/Accept', [], [010.jpeg, 011.jpeg, 012.jpeg, ...]), ('/home/emmaka/Documents/Programming/comp-asl/final/SLR/train_frames/Away', [], [010, 011, 012, ...]), ... ]
    
    '''
    frames = [i for i in os.walk(frames_folder) if os.path.basename(i[0]) in labels_in_dir]

    predictions = []
    for each in frames:
        label = each[0]
        print("Predicting on frame of %s\n" % (label))
        for i in tqdm(range(0, len(each[2]), batch_size), ascii=True):
            batch = each[2][i:i + batch_size] # batch: [010.jpeg, 011.jpeg, 012.jpeg] for each iteration.

            try:
                # batch: ['training_frames/away/010.jpeg', 'training_frames/away/011.jpeg', ...]
                batch = [os.path.join(label, frame) for frame in batch]
                #frame
                frames_tensors = get_tensor_from_image(batch, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)
                prediction = predict(graph, frames_tensors, input_layer, output_layer)
                prediction = [[i.tolist(), os.path.basename(label)] for i in prediction]
                predictions.extend(prediction)

            except KeyboardInterrupt:
                print("You quit with ctrl+c")
                sys.exit()

            except Exception as e:
                print("Error making prediction: %s" % (e))
                x = input("\nContinue on other samples: y/n:")
                if x.lower() == 'y':
                    continue
                else:
                    sys.exit()
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="graph/model to be executed")
    parser.add_argument("frames_folder", help="'Path to folder containing folders of frames of different gestures.'")
    parser.add_argument("--input_layer", help="name of input layer", default='Placeholder')
    parser.add_argument("--output_layer", help="name of output layer", default='final_result')
    parser.add_argument('--test', action='store_true', help='passed if frames_folder belongs to test_data')
    parser.add_argument("--batch_size", help="batch Size", default=10)
    args = parser.parse_args()

    model_file = args.graph
    frames_dir = args.frames_folder
    input_layer = args.input_layer
    output_layer = args.output_layer
    batch_size = int(args.batch_size)


    # if not explicitly mentioned test, consider the frame folder as Training  data.
    if args.test:
        train_or_test = "test"
    else:
        train_or_test = "train"

    # reduce tf verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    predictions = predict_on_frames(frames_dir, model_file, input_layer, output_layer, batch_size)

    output_file = 'predicted-frames-%s-%s.pkl' % (output_layer.split("/")[-1], train_or_test)
    print("Dumping predictions to: %s" % (output_file))
    with open(output_file, 'wb') as fout:
        pickle.dump(predictions, fout)

    print("Done.")
