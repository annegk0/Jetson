
import numpy as np
import time
import csv

import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import torch
from torch import nn


import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

input_data= np.load('/home/brain/Documents/AudioNN_Portage/MobileNetTens/image_701x64.npy') # attention ! à modifier en cas de changement de fichier 
classes_name = np.load("/home/brain/Documents/AudioNN_Portage/MobileNetTens/classes.npy")


# Load trt engine file
model_path_trt = "/home/brain/Documents/Test/audioset_tagging_cnn/TensorRT/mobileNet_engine.trt"

with open(model_path_trt, 'rb') as f:
	engine_data = f.read()
	
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

	
# Allocate input and output buffers on the GPU
input_buf = cuda.mem_alloc(input_data.nbytes)
print(input_buf)


output_shape = (526,)  # Example output shape
output_buf = cuda.mem_alloc(output_shape[0] * np.dtype(np.float32).itemsize)

# Transfer input data to GPU
cuda.memcpy_htod(input_buf, input_data)

# Run inference
context.execute(bindings=[int(input_buf),int(output_buf)])

# Transfer output data from GPU
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, output_buf)

# Post-process output data (adjust based on your model's output)
# For example, you might print the top predicted class and its probability:

# Print Result 

framewise_output = output_data
sorted_indexes = np.argsort(framewise_output)[::-1]

top_k = 10  # Show top results
top_result_mat = framewise_output[sorted_indexes[0 : top_k]]    
top_classes = classes_name[sorted_indexes[0 : top_k]]
for i in range(top_k):
    print(i+1, "result : ", top_result_mat[i], " with class : ",top_classes[i],"\n")





