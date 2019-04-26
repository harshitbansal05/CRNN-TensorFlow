import os
import os.path as ops
import glob

import tensorflow as tf


class CrnnFeatureReader(object):
    def __init__(self, tfrecords_path, batch_size, num_threads):
        self.batch_size = batch_size
        self.dataset = tf.data.TFRecordDataset(tfrecords_path)
        self.dataset = self.dataset.batch(batch_size, drop_remainder=True)
        self.dataset = self.dataset.map(map_func=self._extract_features_batch,
                                        num_parallel_calls=num_threads)
        self.dataset = self.dataset.map(map_func=self._normalize,
                                        num_parallel_calls=num_threads)

        self.dataset = self.dataset.shuffle(buffer_size=1000)
        self.dataset = self.dataset.repeat()

        self.iterator = self.dataset.make_one_shot_iterator()

    def _extract_features_batch(self, serialized_batch):
        features = tf.parse_example(
            serialized_batch,
            features={'images': tf.FixedLenFeature([], tf.string),
                      'imagepaths': tf.FixedLenFeature([], tf.string),
                      'labels': tf.VarLenFeature(tf.int64),
                      }
        )
        images = tf.decode_raw(features['images'], tf.uint8)
        w, h = 100, 32
        images = tf.cast(x=images, dtype=tf.float32)
        images = tf.reshape(images, [self.batch_size, h, w, 3])

        labels = features['labels']
        labels = tf.cast(labels, tf.int32)

        imagepaths = features['imagepaths']

        return images, labels, imagepaths

    def _normalize(self, input_images, input_labels, input_image_paths):
        input_images = tf.subtract(tf.divide(input_images, 127.5), 1.0)
        return input_images, input_labels, input_image_paths

    def inputs(self):
        return self.iterator.get_next(name='train_iterator_get_next')


class CrnnDataFeeder(object):
    def __init__(self, dataset_dir, batch_size):
        tfrecords_file_paths = glob.glob('{:s}/train_*.tfrecords'.format(dataset_dir))
        self._tfrecords_reader = CrnnFeatureReader(
            tfrecords_path=tfrecords_file_paths,
            batch_size=batch_size,
            num_threads=4
        )

    def inputs(self):
        return self._tfrecords_reader.inputs()
