import tensorflow as tf

tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

tf.test.gpu_device_name()
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print('DONE')