import tensorflow as tf

dataset = tf.data.TFRecordDataset(["presentationData.tfrecords"])
print(dataset)