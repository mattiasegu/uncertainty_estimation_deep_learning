# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import print_function
import os
import sys
import tensorflow as tf


def datafiles(search_dir, name):
    tf_record_pattern = os.path.join(search_dir, '%s-*' % name)
    data_files = tf.gfile.Glob(tf_record_pattern)
    data_files = sorted(data_files)
    if not data_files:
      print('No files found for dataset %s at %s' % (name, search_dir))
    return data_files


def example_parser(example_serialized):
    
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/timestamp': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'steer/angle': tf.FixedLenFeature([2], dtype=tf.float32, default_value=[0.0, 0.0]),
        'steer/timestamp': tf.FixedLenFeature([2], dtype=tf.int64, default_value=[-1, -1]),
        #'gps/lat': tf.FixedLenFeature([2], dtype=tf.float32, default_value=[0.0, 0.00]),
        #'gps/long': tf.FixedLenFeature([2], dtype=tf.float32, default_value=[0.0, 0.0]),
        #'gps/timestamp': tf.VarLenFeature(tf.int64),
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    image_timestamp = tf.cast(features['image/timestamp'], dtype=tf.int64)
    steering_angles = features['steer/angle']
    steering_timestamps = features['steer/timestamp']

    return features['image/encoded'], image_timestamp, steering_angles, steering_timestamps


def create_read_graph(data_dir, name, num_readers=4, estimated_examples_per_shard=64, coder=None):
    # Get sharded tf example files for the dataset
    data_files = datafiles(data_dir, name)

    # Create queue for sharded tf example files
    # FIXME the num_epochs argument seems to have no impact? Queue keeps looping forever if not stopped.
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1, num_epochs=1)

    # Create queue for examples
    examples_queue = tf.FIFOQueue(capacity=estimated_examples_per_shard + 4, dtypes=[tf.string])

    enqueue_ops = []
    processed = []
    if num_readers > 1:
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, example = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([example]))
        example_serialized = examples_queue.dequeue()
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
    else:
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

    for x in range(10):
        image_buffer, image_timestamp, steering_angles, steering_timestamps = example_parser(example_serialized)
        decoded_image = tf.image.decode_jpeg(image_buffer)
        print(decoded_image.get_shape(), image_timestamp.get_shape(), steering_angles.get_shape(), steering_timestamps.get_shape())
        decoded_image = tf.reshape(decoded_image, shape=[480, 640, 3])
        processed.append((decoded_image, image_timestamp, steering_angles, steering_timestamps))

    batch_size = 10
    batch_queue_capacity = 2 * batch_size
    batch_data = tf.train.batch_join(
        processed,
        batch_size=batch_size,
        capacity=batch_queue_capacity)

    return batch_data


def main():
    data_dir = '/output/combined'
    num_images = 1452601

    # Build graph and initialize variables
    read_op = create_read_graph(data_dir, 'combined')
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)

    # Start input enqueue threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    read_count = 0
    try:
        while read_count < num_images and not coord.should_stop():
            images, timestamps, angles, _ = sess.run(read_op)
            for i in range(images.shape[0]):
                decoded_image = images[i]
                assert decoded_image.shape[2] == 3
                print(angles[i])
                read_count += 1
            if not read_count % 1000:
                print("Read %d examples" % read_count)

    except tf.errors.OutOfRangeError:
        print("Reading stopped by Queue")
    finally:
        # Ask the threads to stop.
        coord.request_stop()

    print("Done reading %d images" % read_count)

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()