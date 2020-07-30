#import tensorflow as tf
from tensorflow.keras.models import load_model

import os
os.environ['TF_KERAS'] = '1'
import keras2onnx
import onnxruntime
import argparse
import inputLayerReplace

def convertKerasToOnnx(fileName):
    model = load_model(fileName)

    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name)

    #onnx_model_file = './model/mask_recognition_v4_updated.onnx'
    fileName, fileExtension = os.path.splitext(fileName)
    onnx_model_file = fileName + ".onnx"
    keras2onnx.save_model(onnx_model, onnx_model_file)
    sess = onnxruntime.InferenceSession(onnx_model_file)

def main(fileName):
    #fileName = "./model/mask_recognition_v4_updated.h5"

    convertKerasToOnnx(fileName)

if __name__=="__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='/model/mask_recognition_v4_updated.h5', help='Model input shape.')
    args = parser.parse_args()

    args = parser.parse_args()
    if (args.model_file == '/model/mask_recognition_v4_updated.h5'):
        modelFilePath = os.getcwd() + args.model_file
    else:
        modelFilePath = args.model_file

    main(modelFilePath)
