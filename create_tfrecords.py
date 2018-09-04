# Borrowed from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md
# Run 'python create_tfrecords.py --output_path data/kitti_train.record'
from __future__ import print_function

import tensorflow as tf
from utils import dataset_util
import pandas as pd
from PIL import Image
import os
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

def create_tf_example(df, img_id):
    image_path = 'data/images/'
    filename = '%06d.png'%img_id
    
    image_path = os.path.join(image_path, filename)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
        
    filename = filename.encode()
    image = Image.open(image_path)
    width, height = image.size
    del image
    
    image_format = b'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    for bbox in np.array(df[['XMin', 'XMax', 'YMin', 'YMax']]):
        xmins += [bbox[0]/width]
        xmaxs += [bbox[1]/width]
        ymins += [bbox[2]/height]
        ymaxs += [bbox[3]/height]
    
    classes_text = [b'Car']*len(df)
    classes = [1]*len(df)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    for img_id in range(7481):
        df = pd.read_csv('data/labels/%06d.csv'%img_id)
        if df.empty:
            continue
        tf_example = create_tf_example(df, img_id)
        writer.write(tf_example.SerializeToString())
        print('\rprocessing %d of all %d images'%(img_id+1, 7481), end="")
    print('\nDone!')
    writer.close()

if __name__ == '__main__':
    tf.app.run()
