# keras31_gpu_test.py

import tensorflow as tf 
print(tf.__version__)   # 2.7.3

gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if gpus:
    print('GPU 사용 가능')
else:
    print('GPU 사용 불가능')
