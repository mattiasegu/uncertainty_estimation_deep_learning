# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import cv2
import imghdr
import heapq
import argparse
import numpy as np
import tensorflow as tf

from bagutils import *


def feature_int64(value_list):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def feature_float(value_list):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def feature_bytes(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_bytes_list(value_list, skip_convert=False):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


def str2float(string):
    try:
        f = float(string)
        return f, True
    except ValueError:
        return 0.0, False


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def check_image_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def to_steering_dict(sample_list=[], target_sample_count=0):
    if not sample_list:
        count = 1 if not target_sample_count else target_sample_count
        return {
            'steer/timestamp': feature_int64([0] * count),
            'steer/angle': feature_float([0.0] * count),
            'steer/torque': feature_float([0.0] * count),
            'steer/speed': feature_float([0.0] * count),
        }
    #  extend samples to target count if set, needed if fixed sized tensors expected on read
    if target_sample_count and len(sample_list) < target_sample_count:
        sample_list += [sample_list[-1]] * (target_sample_count - len(sample_list))
    timestamps = []
    angles = []
    torques = []
    speeds = []
    for timestamp, msg in sample_list:
        timestamps += [timestamp]
        angles += [msg.steering_wheel_angle]
        torques += [msg.steering_wheel_torque]
        speeds += [msg.speed]
    steering_dict = {
        'steer/timestamp': feature_int64(timestamps),
        'steer/angle': feature_float(angles),
        'steer/torque': feature_float(torques),
        'steer/speed': feature_float(speeds),
    }
    return steering_dict


def to_gps_dict(sample_list=[], target_sample_count=0):
    if not sample_list:
        count = 1 if not target_sample_count else target_sample_count
        return {
            'gps/timestamp': feature_int64([0] * count),
            'gps/lat': feature_float([0.0] * count),
            'gps/long': feature_float([0.0] * count),
            'gps/alt': feature_float([0.0] * count),
        }
    #  extend samples to target count if set, needed if fixed sized tensors expected on read
    if target_sample_count and len(sample_list) < target_sample_count:
        sample_list += [sample_list[-1]] * (target_sample_count - len(sample_list))
    timestamps = []
    lats = []
    longs = []
    alts = []
    for timestamp, msg in sample_list:
        timestamps += [timestamp]
        lats += [msg.latitude]
        longs += [msg.longitude]
        alts += [msg.altitude]
    gps_dict = {
        'gps/timestamp': feature_int64(timestamps),
        'gps/lat': feature_float(lats),
        'gps/long': feature_float(longs),
        'gps/alt': feature_float(alts),
    }
    return gps_dict


class ShardWriter():
    def __init__(self, outdir, name, num_entries, max_num_shards=256):
        self.num_entries = num_entries
        self.outdir = outdir
        self.name = name
        self.max_num_shards = max_num_shards
        self.num_entries_per_shard = num_entries // max_num_shards
        self.write_counter = 0
        self._shard_counter = 0
        self._writer = None

    def _update_writer(self):
        if not self._writer or self._shard_counter >= self.num_entries_per_shard:
            shard = self.write_counter // self.num_entries_per_shard
            assert(shard <= self.max_num_shards)
            output_filename = '%s-%.5d-of-%.5d' % (self.name, shard, self.max_num_shards-1)
            output_file = os.path.join(self.outdir, output_filename)
            self._writer = tf.python_io.TFRecordWriter(output_file)
            self._shard_counter = 0

    def write(self, example):
        self._update_writer()
        self._writer.write(example.SerializeToString())
        self._shard_counter += 1
        self.write_counter += 1
        if not self.write_counter % 1000:
            print('Written %d of %d images for %s' % (self.write_counter, self.num_entries, self.name))
            sys.stdout.flush()


# FIXME lame constants
MIN_SPEED = 1.0  # 2 m/s ~ 8km/h ~ 5mph
WRITE_ENABLE_SLOW_START = 10  # 10 steering samples above min speed before restart


def dequeue_samples_until(queue, timestamp):
    samples = []
    while queue and queue[0][0] < timestamp:
        samples.append(heapq.heappop(queue))
    return samples


class Processor(object):

    def __init__(self,
                 save_dir,
                 num_images,
                 splits=('train', 1.0),
                 name='records',
                 image_fmt='jpg',
                 center_only=False,
                 debug_print=False):

        # config and helpers
        self.debug_print = debug_print
        self.min_buffer_ns = 240 * SEC_PER_NANOSEC  # keep x sec of sorting/sync buffer as per image timestamps
        self.steering_offset_ns = 0  # shift steering timestamps by this much going into queue FIXME test
        self.gps_offset_ns = 0  # shift gps timestamps by this much going into queue FIXME test
        self.bridge = CvBridge()

        # example fixed write params
        self.write_image_fmt = image_fmt
        self.write_colorspace = b'RGB'
        self.write_channels = 3

        # setup writer
        # at approx 35-40KB per image, 6K per shard gives around 200MB per shard
        #FIXME maybe support splitting data stream into train/validation from the same bags?
        num_shards = num_images // 6000
        self._outdir = get_outdir(save_dir, name)
        self._writers = {}
        for s in splits:
            scaled_images = num_images * s[1]
            scaled_shards = num_shards * s[1]
            if s[0] == 'validation' and not center_only:
                scaled_images //= 3
                scaled_shards //= 3
            writer = ShardWriter(self._outdir, s[0], scaled_images, max_num_shards=scaled_shards)
            self._writers[s[0]] = writer
        self._splits = splits

        # stats, counts, and queues
        self.written_image_count = 0
        self.discarded_image_count = 0
        self.collect_image_stats = False
        self.collect_io_stats = True
        self.image_means = []
        self.image_variances = []
        self.steering_vals = []
        self.gps_vals = []
        self.reset_queues()

    def _select_writer(self):
        r = np.random.random_sample()
        for s in self._splits:
            if r < s[1]:
                return self._writers[s[0]]
            r -= s[1]
        return None

    def reset_queues(self):
        self.latest_image_timestamp = None
        self._write_enable = False
        self._speed_above_min_count = 0
        self._steering_queue = []   # time sorted steering heap
        self._gear_queue = [] # time sorted gear heap
        self._gps_queue = []  # time sorted gps heap
        self._images_queue = []  # time sorted image heap
        self._head_gear_sample = None
        self._head_steering_sample = None  # most recent steering timestamp/topic/msg sample pulled from queue
        self._head_gps_sample = None  # most recent gps timestamp/topic/msg sample pulled from queue
        self._debug_gps_next = False

    def write_example(self, image_topic, image_msg, steering_list, gps_list, dataset_id=0):
        try:
            assert isinstance(steering_list, list)
            assert isinstance(gps_list, list)
            
            writer = self._select_writer()
            if writer is None:
                self.discarded_image_count += 1
                return 
            elif writer.name == 'validation':
                if image_topic not in CENTER_CAMERA_TOPICS:
                    self.discarded_image_count += 1
                    return

            image_width = 0
            image_height = 0
            if hasattr(image_msg, 'format') and 'compressed' in image_msg.format:
                buf = np.ndarray(shape=(1, len(image_msg.data)), dtype=np.uint8, buffer=image_msg.data)
                cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
                if cv_image.shape[2] != 3:
                    print("Invalid image")
                    return
                image_height = cv_image.shape[0]
                image_width = cv_image.shape[1]
                # Avoid re-encoding if we don't have to
                if check_image_format(image_msg.data) == self.write_image_fmt:
                    encoded = buf
                else:
                    _, encoded = cv2.imencode('.' + self.write_image_fmt, cv_image)
            else:
                image_width = image_msg.width
                image_height = image_msg.height
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
                _, encoded = cv2.imencode('.' + self.write_image_fmt, cv_image)

            if self.collect_image_stats:
                mean, std = cv2.meanStdDev(cv_image)
                self.image_means.append(np.squeeze(mean))
                self.image_variances.append(np.squeeze(np.square(std)))

            if self.collect_io_stats:
                self.steering_vals.extend([x[1].steering_wheel_angle for x in steering_list])
                self.gps_vals.extend([[x[1].latitude, x[1].longitude] for x in gps_list])

            feature_dict = {
                'image/timestamp': feature_int64(image_msg.header.stamp.to_nsec()),
                'image/frame_id': feature_bytes(image_msg.header.frame_id),
                'image/height': feature_int64(image_height),
                'image/width': feature_int64(image_width),
                'image/channels': feature_int64(self.write_channels),
                'image/colorspace': feature_bytes(self.write_colorspace),
                'image/format': feature_bytes(self.write_image_fmt),
                'image/encoded': feature_bytes(encoded.tobytes()),
                'image/dataset_id': feature_int64(dataset_id),
            }
            steering_dict = to_steering_dict(steering_list, target_sample_count=2)
            feature_dict.update(steering_dict)
            gps_dict = to_gps_dict(gps_list, target_sample_count=2)
            feature_dict.update(gps_dict)
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example)
            self.written_image_count += 1

        except CvBridgeError as e:
            print(e)

    def push_messages(self, messages):
        for timestamp, topic, msg in messages:
            if topic in CAMERA_TOPICS:
                heapq.heappush(self._images_queue, (timestamp, topic, msg))
                if not self.latest_image_timestamp or timestamp > self.latest_image_timestamp:
                    self.latest_image_timestamp = timestamp
            elif topic == STEERING_TOPIC:
                if self.debug_print:
                    print("%s: steering, %f" % (ns_to_str(timestamp), msg.steering_wheel_angle))
                timestamp += self.steering_offset_ns
                heapq.heappush(self._steering_queue, (timestamp, msg))
            elif topic == GEAR_TOPIC:
                timestamp += self.steering_offset_ns # same offset as steering
                heapq.heappush(self._gear_queue, (timestamp, msg))
            elif topic == GPS_FIX_TOPIC or topic == GPS_FIX_NEW_TOPIC:
                if self._debug_gps_next or self.debug_print:
                    print("%s: gps     , (%f, %f)" % (ns_to_str(timestamp), msg.latitude, msg.longitude))
                    self._debug_gps_next = False
                timestamp += self.gps_offset_ns
                heapq.heappush(self._gps_queue, (timestamp, msg))

    def _update_write_enable(self, image_timestamp, steering_samples, latest_gear_sample):
        gear_forward = False if latest_gear_sample and latest_gear_sample[1].state.gear <= 2 else True
        gear = latest_gear_sample[1].state.gear if latest_gear_sample else 0
        for sample in steering_samples:
            sample_speed = sample[1].speed
            if self._write_enable:
                if sample_speed < MIN_SPEED or not gear_forward:
                    # disable writing instantly on sample below minimum
                    self._write_enable = False
                    print('%s: Write disable. Speed: %s, gear: %d '
                          % (ns_to_str(image_timestamp), sample_speed, gear))
                    self._speed_above_min_count = 0
                else:
                    self._speed_above_min_count += 1
            else:  # not write enable
                # enable writing after threshold number samples above minimum seen
                if sample_speed < MIN_SPEED or not gear_forward:
                    self._speed_above_min_count = 0
                else:
                    self._speed_above_min_count += 1
                if self._speed_above_min_count > WRITE_ENABLE_SLOW_START:
                    self._write_enable = True
                    print('%s: Write enable. Speed: %s, gear: %d'
                          % (ns_to_str(image_timestamp), sample_speed, gear))

    def pull_and_write(self, flush=False):
        while self.pull_ready(flush):
            assert self._images_queue
            image_timestamp, image_topic, image_msg = heapq.heappop(self._images_queue)
            if self.debug_print:
                print("Popped image: %d, %s" % (image_timestamp, image_topic))

            gear_samples = dequeue_samples_until(self._gear_queue, image_timestamp)
            if gear_samples:
                self._head_gear_sample = gear_samples[-1]

            steering_samples = dequeue_samples_until(self._steering_queue, image_timestamp)
            if steering_samples:
                self._head_steering_sample = steering_samples[-1]
                self._update_write_enable(image_timestamp, steering_samples, self._head_gear_sample)

            gps_samples = dequeue_samples_until(self._gps_queue, image_timestamp)
            if gps_samples:
                self._head_gps_sample = gps_samples[-1]

            if self._write_enable:
                steering_list = []
                gps_list = []
                if self._head_steering_sample:
                    steering_list.append(self._head_steering_sample)
                else:
                    print('%s: Invalid head steering sample!' % ns_to_str(image_timestamp))
                if self._steering_queue:
                    steering_list.append(self._steering_queue[0])
                else:
                    print('%s: Empty steering queue!' % ns_to_str(image_timestamp))
                if self._head_gps_sample:
                    gps_list.append(self._head_gps_sample)
                else:
                    print('%s: Invalid head gps sample!' % ns_to_str(image_timestamp))
                    self._debug_gps_next = True
                if self._gps_queue:
                    gps_list.append(self._gps_queue[0])
                else:
                    print('%s: Empty gps queue!' % ns_to_str(image_timestamp))
                    self._debug_gps_next = True

                self.write_example(image_topic, image_msg, steering_list, gps_list)
            else:
                self.discarded_image_count += 1

    def _remaining_time(self):
        if not self._images_queue:
            return 0
        return self.latest_image_timestamp - self._images_queue[0][0]

    def pull_ready(self, flush=False):
        return self._images_queue and (flush or self._remaining_time() > self.min_buffer_ns)

    def get_writer_counts(self):
        counts = []
        for w in self._writers.values():
            counts.append((w.name, w.write_counter))
        return counts


def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to tensorflow sharded records.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-b', '--indir', type=str, nargs='?', default='/data/',
        help='Input bag file')
    parser.add_argument('-f', '--image_fmt', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-s', '--split', type=str, nargs='?', default='train',
        help="Data subset. 'train' or 'validation'")
    parser.add_argument('-k', '--keep', type=float, nargs='?', default=1.0,
        help="Keep specified percent of data. 0.0 or 1.0 is all")
    parser.add_argument('-c', '--center', action='store_true',
        help="Center camera only for all splits")
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(center=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    image_fmt = args.image_fmt
    save_dir = args.outdir
    input_dir = args.indir
    debug_print = args.debug
    center_only = args.center
    split = args.split
    keep = args.keep
    if keep == 0.0 or keep > 1.0:
        # 0 is invalid, change to keep all
        keep = 1.0

    filter_topics = [STEERING_TOPIC, GPS_FIX_TOPIC, GPS_FIX_NEW_TOPIC, GEAR_TOPIC]
    split_val, is_float = str2float(split)
    if is_float and 0.0 < split_val < 1.0:
        # split specified as float val indicating %validation data
        filter_camera_topics = CAMERA_TOPICS if not center_only else CENTER_CAMERA_TOPICS
        split_val *= keep
        split_list = [('train', keep - split_val), ('validation', split_val)]
    elif split == 'validation' or (is_float and split_val == 1.0):
        # split specified to be validation, set as 100% validation
        filter_camera_topics = CENTER_CAMERA_TOPICS
        split_list = [(split, keep)]
    else:
        # 100% train split
        assert split == 'train'
        filter_camera_topics = CAMERA_TOPICS if not center_only else CENTER_CAMERA_TOPICS
        split_list = [(split, keep)]
    filter_topics += filter_camera_topics

    num_images = 0
    num_messages = 0
    bagsets = find_bagsets(input_dir, filter_topics=filter_topics)
    for bs in bagsets:
        num_images += bs.get_message_count(filter_camera_topics)
        num_messages += bs.get_message_count(filter_topics)
    print("%d images, %d messages to import across %d bag sets..."
          % (num_images, num_messages, len(bagsets)))

    processor = Processor(
        save_dir=save_dir, num_images=num_images, image_fmt=image_fmt,
        splits=split_list, center_only=center_only, debug_print=debug_print)

    num_read_messages = 0  # number of messages read by cursors
    aborted = False
    try:
        for bs in bagsets:
            print("Processing set %s. %s to %s" % (bs.name, ns_to_str(bs.start_time), ns_to_str(bs.end_time)))
            sys.stdout.flush()

            cursor_group = CursorGroup(readers=bs.get_readers())
            while cursor_group:
                msg_tuples = []
                cursor_group.advance_by_until(360 * SEC_PER_NANOSEC)
                cursor_group.collect_vals(msg_tuples)
                num_read_messages += len(msg_tuples)
                processor.push_messages(msg_tuples)
                if processor.pull_ready():
                    processor.pull_and_write()

            processor.pull_and_write(flush=True)  # flush remaining messages after read cursors are done
            processor.reset_queues()  # ready for next bag set
    except KeyboardInterrupt:
        aborted = True

    if not aborted:
        if num_read_messages != num_messages:
            print("Number of read messages (%d) doesn't match expected count (%d)" %
                  (num_read_messages, num_messages))
        total_processed_images = processor.written_image_count + processor.discarded_image_count
        if total_processed_images != num_images:
            print("Number of processed images (%d) doesn't match expected count (%d)" %
                  (total_processed_images, num_images))

    print("Completed processing %d images to TF examples. %d images discarded" %
          (processor.written_image_count, processor.discarded_image_count))

    print("Writer counts: ")
    [print("\t%s: %d" % (x[0], x[1])) for x in processor.get_writer_counts()]

    if processor.collect_image_stats:
        channel_mean = np.mean(processor.image_means, axis=0, dtype=np.float64)[::-1]
        channel_std = np.sqrt(np.mean(processor.image_variances, axis=0, dtype=np.float64))[::-1]
        print("Mean: ", channel_mean, ". Std deviation: ", channel_std)

    if processor.collect_io_stats:
        steering_mean = np.mean(processor.steering_vals, axis=0, dtype=np.float64)
        steering_std = np.std(processor.steering_vals, axis=0, dtype=np.float64)
        steering_min = np.min(processor.steering_vals, axis=0)
        steering_max = np.max(processor.steering_vals, axis=0)
        gps_mean = np.mean(processor.gps_vals, axis=0, dtype=np.float64)
        gps_std = np.std(processor.gps_vals, axis=0, dtype=np.float64)
        gps_min = np.min(processor.gps_vals, axis=0)
        gps_max = np.max(processor.gps_vals, axis=0)
        print("Steering: ")
        print("\tmean: ", steering_mean)
        print("\tstd: ", steering_std)
        print("\tmin: ", steering_min)
        print("\tmax: ", steering_max)
        print("Gps: ")
        print("\tmean: ", gps_mean)
        print("\tstd: ", gps_std)
        print("\tmin: ", gps_min)
        print("\tmax: ", gps_max)

    sys.stdout.flush()


if __name__ == '__main__':
    main()