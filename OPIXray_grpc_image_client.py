#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import partial
import argparse
import numpy as np
import time
import os
import sys

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

sys.path.append("./OPIXray/DOAM")
import torch
from OPIXray.DOAM.layers.functions.detection import Detect
from OPIXray.DOAM.detection_draw import draw_with_coordinate
sys.path.append("./OPIXray/DOAM/utils")
from OPIXray.DOAM.utils.predict_struct import result_struct
from OPIXray.DOAM.data import OPIXray_CLASSES
from PIL import Image
import cv2

def preprocess(img, h, w):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    resized_img = cv2.resize(img,(w, h))
    #  RGB2BGR
    resized = np.array(resized_img)
    resized = resized.transpose(2, 0, 1)[np.newaxis,::].astype(np.float32)
    print(resized.shape,resized.dtype)
    return resized


def file_paser(FLAGS,h=300,w=300):
    filenames = []
    if os.path.isdir(FLAGS.image_filename):
        filenames = [
            os.path.join(FLAGS.image_filename, f)
            for f in os.listdir(FLAGS.image_filename)
            if os.path.isfile(os.path.join(FLAGS.image_filename, f))
        ]
    else:
        filenames = [
            FLAGS.image_filename,
        ]

    filenames.sort()

    image_data = []
    og_ims = []
    for filename in filenames:
        img = cv2.imread(filename)
        og_ims.append(img)
        image_data.append(preprocess(img, h, w,))
    return image_data,og_ims


def plot_result(detections,og_im,h=954,w=1225,classes=OPIXray_CLASSES):
    all_boxes = [[[] for _ in range(1)]
                 for _ in range(len(classes) + 1)]
    class_correct_scores, class_coordinate_dict = result_struct(detections, h, w, all_boxes=all_boxes, OPIXray_CLASSES=OPIXray_CLASSES)
    print(class_coordinate_dict)
    draw_with_coordinate(class_correct_scores, class_coordinate_dict,og_im)

if __name__ == '__main__':
    # python OPIXray_grpc_image_client.py  -u 192.168.8.187:8001 -m opi
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-m',
                        '--model-name',
                        type=str,
                        required=True,
                        help='Name of model')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds. Default is None.')
    parser.add_argument('image_filename',
                        type=str,
                        nargs='?',
                        default=None,
                        help='Input image / Input folder.')

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    model_name = FLAGS.model_name

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('modelInput', [1,3, 300,300], "FP32"))
    # inputs.append(grpcclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.

    # input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    # input0_data = np.expand_dims(input0_data, axis=0)

    image_data,og_ims = file_paser(FLAGS)
    # input0_data = np.ones(shape=(1,3,300,300), dtype=np.float32)
    input0_data = image_data[0]
    # Initialize the data

    inputs[0].set_data_from_numpy(input0_data)

    outputs.append(grpcclient.InferRequestedOutput('modelOutput'))
    outputs.append(grpcclient.InferRequestedOutput('407'))
    outputs.append(grpcclient.InferRequestedOutput('408'))

    # Define the callback function. Note the last two parameters should be
    # result and error. InferenceServerClient would povide the results of an
    # inference as grpcclient.InferResult in result. For successful
    # inference, error will be None, otherwise it will be an object of
    # tritonclientutils.InferenceServerException holding the error details
    def callback(user_data, result, error):
        if error:
            user_data.append(error)
        else:
            user_data.append(result)

    # list to hold the results of inference.
    user_data = []

    # Inference call
    triton_client.async_infer(model_name=model_name,
                              inputs=inputs,
                              callback=partial(callback, user_data),
                              outputs=outputs,
                              client_timeout=FLAGS.client_timeout)
    start1 = time.time()
    # Wait until the results are available in user_data
    time_out = 10
    while ((len(user_data) == 0) and time_out > 0):
        time_out = time_out - .1
        time.sleep(.1)



    # Display and validate the available results
    print((len(user_data)))
    if ((len(user_data) == 1)):
        # Check for the errors
        if type(user_data[0]) == InferenceServerException:
            print(user_data[0])
            sys.exit(1)

        # Validate the values by matching with already computed expected
        # values.
        # outputs.append(grpcclient.InferRequestedOutput('264'))
        # outputs.append(grpcclient.InferRequestedOutput('modelOutput'))
        # outputs.append(grpcclient.InferRequestedOutput('406'))

        output0_data = torch.from_numpy(user_data[0].as_numpy('modelOutput'))
        output1_data = torch.from_numpy(user_data[0].as_numpy('407'))
        output2_data = torch.from_numpy(user_data[0].as_numpy('408'))
        print(output0_data.shape,output1_data.shape,output2_data.shape)
        # exit(0)
        detect = Detect(6, 0, 200, 0.01, 0.45)
        result = detect.forward(output0_data,output1_data,output2_data).data

        print(time.time() - start1, "s")
        plot_result(result,og_im=og_ims[0])
        # for i in range(16):
        #     print(
        #         str(input0_data[0][i]) + " + " + str(input1_data[0][i]) +
        #         " = " + str(output0_data[0][i]))
        #     print(
        #         str(input0_data[0][i]) + " - " + str(input1_data[0][i]) +
        #         " = " + str(output1_data[0][i]))
        #     if (input0_data[0][i] + input1_data[0][i]) != output0_data[0][i]:
        #         print("sync infer error: incorrect sum")
        #         sys.exit(1)
        #     if (input0_data[0][i] - input1_data[0][i]) != output1_data[0][i]:
        #         print("sync infer error: incorrect difference")
        #         sys.exit(1)
        print("PASS: Async infer")