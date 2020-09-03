import tensorflow as tf


def open_tfrecord():
    filenames = ['bach-doodle.tfrecord-00107-of-00192']
    raw_dataset = tf.data.TFRecordDataset(filenames)
    print(raw_dataset)

    for raw_record in raw_dataset:
        example = tf.train.Example()
        example.ParseFromString(raw_record.string())
        print(example)


if __name__ == '__main__':
    open_tfrecord()