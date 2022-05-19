import os
import cv2
import glob
import logging
import traceback
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import ImageFile
import numpy as np
import argparse
import sys
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
from OPIXray_grpc_image_client import *
#from OPIXray.DOAM.detection_draw import draw_with_coordinate_dynamic

# different color
# COLOR_CONFIG = {
#     'Folding_Knife': (0, 0, 0)
#     , 'Straight_Knife': (0,100,0)
#     , 'Scissor': (0,0,128)
#     , 'Utility_Knife': (255, 0, 255)
#     , 'Multi-tool_Knife': (255, 0, 0),
# }

# Purple
# COLOR_CONFIG = {
#     'Folding_Knife': (72,61,139)
#     , 'Straight_Knife': (72,61,139)
#     , 'Scissor': (72,61,139)
#     , 'Utility_Knife': (72,61,139)
#     , 'Multi-tool_Knife': (72,61,139),
# }

# pink
COLOR_CONFIG = {
    'Folding_Knife': (255,0,255)
    , 'Straight_Knife': (255,0,255)
    , 'Scissor': (255,0,255)
    , 'Utility_Knife': (255,0,255)
    , 'Multi-tool_Knife': (255,0,255),
}

# Red
# COLOR_CONFIG = {
#     'Folding_Knife': (255,0,0)
#     , 'Straight_Knife': (255,0,0)
#     , 'Scissor': (255,0,0)
#     , 'Utility_Knife': (255,0,0)
#     , 'Multi-tool_Knife': (255,0,0),
# }

def draw_with_coordinate_dynamic(class_correct_scores: dict, class_coordinate_dict: dict, og_im,
                                 color_config=COLOR_CONFIG):
    og_im_copy = og_im.copy()
    for cls, scores in class_correct_scores.items():
        if scores:
            for index, score in enumerate(scores):
                coordinate = tuple(map(int, class_coordinate_dict[cls][index]))
                first_point = (coordinate[0], coordinate[1])
                last_point = (coordinate[2], coordinate[3])
                cv2.rectangle(og_im, first_point, last_point, color_config[cls], 2)
                # 在矩形框上方绘制该框的名称
                text_point = ((coordinate[0], coordinate[1] - 4 if coordinate[1] - 4 > 0 else coordinate[1]))
                cv2.putText(og_im, "{0},score:{1}".format(cls, "%.2f" % score), text_point,
                            cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1.5, color=color_config[cls],
                            thickness=3)
    return og_im_copy, og_im


def plot_result_dynamic(detections, og_ims, h=954, w=1225, classes=OPIXray_CLASSES):
    fig, axes = plt.subplots(1, 2, figsize=(12,12), dpi=200)
    plt.ion()
    for i in range(len(detections)):
        all_boxes = [[[] for _ in range(1)]
                     for _ in range(len(classes) + 1)]
        class_correct_scores, class_coordinate_dict = result_struct(detections[i], h, w, all_boxes=all_boxes,
                                                                    OPIXray_CLASSES=OPIXray_CLASSES)
        # print(class_coordinate_dict)
        # draw_with_coordinate(class_correct_scores, class_coordinate_dict,og_im)
        image1, image2 = draw_with_coordinate_dynamic(class_correct_scores, class_coordinate_dict, og_ims[i])

        if i == 0:
            am0 = axes[0].imshow(image1)
            axes[0].set_title("Xray Image (1)",fontsize=20)
            axes[0].axis('off')
            am1 = axes[1].imshow(image2)
            axes[1].set_title("Result (1)",fontsize=20)
            axes[1].axis('off')
        else:
            am0.set_data(image1)
            am1.set_data(image2)
            axes[0].set_title(f"Xray Image ({i+1})",fontsize=20)
            axes[1].set_title(f"Result ({i+1})",fontsize=20)
            fig.canvas.flush_events()
        plt.pause(1)
    plt.ioff()


if __name__ == '__main__':
    # python demo.py  -u 192.168.8.187:230 -m opi
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
    # inputs = []
    # outputs = []
    # inputs.append(grpcclient.InferInput('modelInput', [1, 3, 300, 300], "FP32"))
    # inputs.append(grpcclient.InferInput('INPUT1', [1, 16], "INT32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.

    # input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    # input0_data = np.expand_dims(input0_data, axis=0)

    image_data, og_ims = file_paser(FLAGS)

    results = []
    # Initialize the data
    for i in range(0,len(image_data)):
        inputs = []
        outputs = []

        inputs.append(grpcclient.InferInput('modelInput', [1, 3, 300, 300], "FP32"))
        inputs[0].set_data_from_numpy(image_data[i])

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
        if ((len(user_data) == 1)):
            results.append(user_data[0])

    print('results length:', (len(results)))
    output_result = []
    for i in range(0,len(results)):
        # Display and validate the available results
        # Check for the errors
        if type(results[i]) == InferenceServerException:
            print(results[i])
            sys.exit(1)

        output0_data = torch.from_numpy(results[i].as_numpy('modelOutput'))
        output1_data = torch.from_numpy(results[i].as_numpy('407'))
        output2_data = torch.from_numpy(results[i].as_numpy('408'))
        print(output0_data.shape, output1_data.shape, output2_data.shape)
        # exit(0)
        detect = Detect(6, 0, 200, 0.01, 0.45)
        result = detect.forward(output0_data, output1_data, output2_data).data
        output_result.append(result)

    print(time.time() - start1, "s")
    plot_result_dynamic(output_result, og_ims=og_ims)

    print("PASS: Async infer")