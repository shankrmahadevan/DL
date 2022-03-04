import argparse
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util
import cv2
import math
import pandas as pd
from tqdm import tqdm
import contextlib2
import tensorflow as tf
from PIL import Image
import io
import os


def create_tf_example(image_path, annotations, class_, labels):
    with tf.io.gfile.GFile(image_path, 'rb') as image_bin:
        image_bin = image_bin.read()
    encoded_jpg_io = io.BytesIO(image_bin)
    image = Image.open(encoded_jpg_io)
    height = image.size[1]
    width = image.size[0]
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    image_format = 'jpeg'.encode('utf8')
    filename = ':'.join(image_path.split('/')[-2:])
    filename = filename.rstrip('.jpg').encode('utf8')
    for index, annotation in enumerate(annotations):
        xmin1 = annotation['x'] / width
        xmax1 = (annotation['x'] + annotation['width']) / width
        ymin1 = annotation['y'] / height
        ymax1 = (annotation['y'] + annotation['height']) / height
        xmins.append(xmin1)
        xmaxs.append(xmax1)
        ymins.append(ymin1)
        ymaxs.append(ymax1)
        classes_text.append(class_[index].encode('utf8'))
        classes.append(labels[index])
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/encoded': dataset_util.bytes_feature(image_bin),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


parser = argparse.ArgumentParser()
parser.add_argument('-image-path', type=str, nargs='?', const='default')
parser.add_argument('-label-file', type=str, nargs='?', const='default')

args = parser.parse_args()
df = pd.read_csv(args.label_file)
base_path = args.image_path

unique = df['label'].unique()
labels = {unique[i]:i+1 for i in range(len(unique))}
def convert_to_tfrecord(df, file_name, num_shards=10):
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, file_name, num_shards)
        for row, file_name in tqdm(enumerate(df['name'].unique())):
          selection = df[df['name'] == file_name]
          annotations_temp, class_name_temp, labels_temp = [], [], []
          for i in range(len(selection)):
            row_temp = selection.iloc[i]
            file_name, x, y, width, height, class_ = row_temp
            annotations_temp.append({'x':x, 'y':y, 'width':width, 'height':height})
            class_name_temp.append(class_)
            labels_temp.append(labels[class_])
          tf_example = create_tf_example(os.path.join(base_path, file_name), annotations_temp, class_name_temp, labels_temp)
          output_shard_index = row % num_shards
          output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

convert_to_tfrecord(df, 'dataset/tfod-dataset')
