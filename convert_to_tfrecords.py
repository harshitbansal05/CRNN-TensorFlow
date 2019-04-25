import os
import os.path as ops
from multiprocessing import Manager
from multiprocessing import Process
import argparse
import pickle
import json

import cv2
import numpy as np
import tensorflow as tf
import tqdm

SAMPLE_INFO_QUEUE = Manager().Queue()
SENTINEL = ("", [])


def _int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, bytes):
        if not isinstance(value, list):
            value = value.encode('utf-8')
        else:
            value = [val.encode('utf-8') for val in value]
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _write_tfrecords(tfrecords_writer):
    while True:
        sample_info = SAMPLE_INFO_QUEUE.get()

        if sample_info == SENTINEL:
            tfrecords_writer.close()
            break

        sample_path = sample_info[0]
        sample_label = sample_info[1]

        try:
            image = cv2.imread(sample_path, cv2.IMREAD_COLOR)
            if image is None:
                print('Image is none')
                continue
            image = cv2.resize(image, dsize=tuple((100, 32)), interpolation=cv2.INTER_LINEAR)
            image = image.tostring()
        except IOError as err:
            print('Exception')
            continue

        features = tf.train.Features(feature={
            'labels': _int64_feature(sample_label),
            'images': _bytes_feature(image),
            'imagepaths': _bytes_feature(sample_path)
        })
        tf_example = tf.train.Example(features=features)
        tfrecords_writer.write(tf_example.SerializeToString())


class CrnnFeatureWriter(object):
    def __init__(self, annotation_infos, lexicon_infos,
                 char_dict, tfrecords_save_dir,
                 writer_process_nums, dataset_flag):
        self._dataset_flag = dataset_flag
        self._annotation_infos = annotation_infos
        self._lexicon_infos = lexicon_infos
        self._char_dict = char_dict
        self._writer_process_nums = writer_process_nums
        self._init_example_info_queue()
        self._tfrecords_save_dir = tfrecords_save_dir

    def _init_example_info_queue(self):
        print('Start filling {:s} dataset sample information queue...'.format(self._dataset_flag))

        for annotation_info in tqdm.tqdm(self._annotation_infos):
            image_path = annotation_info[0]
            lexicon_index = annotation_info[1]

            lexicon_label = self._lexicon_infos[lexicon_index]
            encoded_label = self.encode_label(lexicon_label)
            SAMPLE_INFO_QUEUE.put((image_path, encoded_label))

        for i in range(self._writer_process_nums):
            SAMPLE_INFO_QUEUE.put(SENTINEL)
        print('Complete filling dataset sample information queue[current size: {:d}]'.format(
            SAMPLE_INFO_QUEUE.qsize()
        ))

    def char_to_int(self, char):
        str_key = str(ord(char)) + '_ord'
        return int(self._char_dict[str_key])
    
    def encode_label(self, label):
        encoded_label = [self.char_to_int(char) for char in label]
        return encoded_label

    def run(self):
        print('Start writing TensorFlow records for {:s}...'.format(self._dataset_flag))

        process_pool = []
        tfwriters = []
        for i in range(self._writer_process_nums):
            tfrecords_save_name = '{:s}_{:d}.tfrecords'.format(self._dataset_flag, i + 1)
            tfrecords_save_path = ops.join(self._tfrecords_save_dir, tfrecords_save_name)

            tfrecords_io_writer = tf.python_io.TFRecordWriter(path=tfrecords_save_path)
            process = Process(
                target=_write_tfrecords,
                name='Subprocess_{:d}'.format(i + 1),
                args=(tfrecords_io_writer,)
            )
            process_pool.append(process)
            tfwriters.append(tfrecords_io_writer)
            process.start()

        for process in process_pool:
            process.join()

        print('Finished writing down the TensorFlow records file')


class CrnnDataProducer(object):
    def __init__(self, dataset_dir, writer_process_nums=4):
        
        self._dataset_dir = dataset_dir
        self._train_annotation_file_path = ops.join(dataset_dir, 'annotation_train.txt')
        self._test_annotation_file_path = ops.join(dataset_dir, 'annotation_test.txt')
        self._val_annotation_file_path = ops.join(dataset_dir, 'annotation_val.txt')
        self._lexicon_file_path = ops.join(dataset_dir, 'lexicon.txt')
        self._char_dict_path = 'char_dict.json'
        self._writer_process_nums = writer_process_nums

        self._lexicon_list = []
        self._train_sample_infos = []
        self._test_sample_infos = []
        self._val_sample_infos = []
        self._init_dataset_sample_info()

        self._generate_char_dict()

    def _init_dataset_sample_info(self):
        num_lines = sum(1 for _ in open(self._lexicon_file_path, 'r'))
        with open(self._lexicon_file_path, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file, total=num_lines):
                self._lexicon_list.append(line.rstrip('\r').rstrip('\n'))

        if os.path.isfile('val_sample_list'):
            with open('val_sample_list', 'rb') as fp:
                self._val_sample_infos = pickle.load(fp)
        else:
            num_lines = sum(1 for _ in open(self._val_annotation_file_path, 'r'))
            with open(self._val_annotation_file_path, 'r', encoding='utf-8') as file:
                for line in tqdm.tqdm(file, total=num_lines):
                    image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                    image_path = ops.join(self._dataset_dir, image_name)
                    label_index = int(label_index)
                    self._val_sample_infos.append((image_path, label_index))

            with open('val_sample_list', 'wb') as fp:
                pickle.dump(self._val_sample_infos, fp)

        if os.path.isfile('train_sample_list'):
            with open('train_sample_list', 'rb') as fp:
                self._train_sample_infos = pickle.load(fp)
        else:
            num_lines = sum(1 for _ in open(self._train_annotation_file_path, 'r'))
            with open(self._train_annotation_file_path, 'r', encoding='utf-8') as file:
                for line in tqdm.tqdm(file, total=num_lines):

                    image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                    image_path = ops.join(self._dataset_dir, image_name)
                    label_index = int(label_index)
                    self._train_sample_infos.append((image_path, label_index))

            with open('train_sample_list', 'wb') as fp:
                pickle.dump(self._train_sample_infos, fp)

        if os.path.isfile('test_sample_list'):
            with open('test_sample_list', 'rb') as fp:
                self._test_sample_infos = pickle.load(fp)
        else:
            num_lines = sum(1 for _ in open(self._test_annotation_file_path, 'r'))
            with open(self._test_annotation_file_path, 'r', encoding='utf-8') as file:
                for line in tqdm.tqdm(file, total=num_lines):
                    image_name, label_index = line.rstrip('\r').rstrip('\n').split(' ')
                    image_path = ops.join(self._dataset_dir, image_name)
                    label_index = int(label_index)
                    self._test_sample_infos.append((image_path, label_index))

            with open('test_sample_list', 'wb') as fp:
                pickle.dump(self._test_sample_infos, fp)

    def _generate_char_dict(self):
        char_lexicon_set = set()
        for lexcion in self._lexicon_list:
            for s in lexcion:
                char_lexicon_set.add(s)

        char_lexicon_list = list(char_lexicon_set)
        self._char_dict = {str(ord(c)) + '_ord': str(i) for i, c in enumerate(char_lexicon_list)}
        with open(self._char_dict_path, 'w', encoding='utf-8') as json_f:
            json.dump(self._char_dict, json_f, sort_keys=True, indent=4)

    def generate_tfrecords(self, save_dir):

        print('Generating testing sample tfrecords....')
        
        tfrecords_writer = CrnnFeatureWriter(
            annotation_infos=self._test_sample_infos,
            lexicon_infos=self._lexicon_list,
            char_dict=self._char_dict,
            tfrecords_save_dir=save_dir,
            writer_process_nums=self._writer_process_nums,
            dataset_flag='test'
        )
        tfrecords_writer.run()
        print('Generating testing sample tfrecords complete')

        print('Generating validation sample tfrecords...')
        
        tfrecords_writer = CrnnFeatureWriter(
            annotation_infos=self._val_sample_infos,
            lexicon_infos=self._lexicon_list,
            char_dict=self._char_dict,
            tfrecords_save_dir=save_dir,
            writer_process_nums=self._writer_process_nums,
            dataset_flag='val'
        )
        tfrecords_writer.run()
        print('Generating validation sample tfrecords complete')

        print('Generating training sample tfrecords...')
        
        tfrecords_writer = tf_io_pipline_fast_tools.CrnnFeatureWriter(
            annotation_infos=self._train_sample_infos,
            lexicon_infos=self._lexicon_list,
            char_dict=self._char_dict,
            tfrecords_save_dir=save_dir,
            writer_process_nums=self._writer_process_nums,
            dataset_flag='train'
        )
        tfrecords_writer.run()
        print('Generating training sample tfrecords complete')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='The synth90k dataset directory')
    parser.add_argument('-s', '--save_dir', type=str, help='The generated tfrecords save directory')
    
    return parser.parse_args()


def write_tfrecords(dataset_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    producer = CrnnDataProducer(
        dataset_dir=dataset_dir,
        writer_process_nums=8
    )

    producer.generate_tfrecords(
        save_dir=save_dir
    )


if __name__ == '__main__':
    args = init_args()

    write_tfrecords(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir
    )
