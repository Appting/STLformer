# Liuzhaoxi 2023/11/7 16:16
import tensorflow as tf

if tf.test.is_gpu_available():
    print("GPU is available.")
else:
    print("GPU is not available.")

# import torch
#
# if torch.cuda.is_available():
#     print("GPU is available.")
# else:
#     print("GPU is not available.")


# import tensorflow as tf
# print(tf.test.is_gpu_available())


