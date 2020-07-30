#import tensorflow as tf
#import tensorflow as tf
import argparse
import os
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.layers import Input


def modifiedInputLayer(modelFileName,inputShape):
    face_classification_model = load_model(modelFileName)
    face_classification_model.get_weights()
    face_classification_model.layers.pop(0)
    #print("#######################################################################################")
    #face_classification_model.summary()
    #print("#######################################################################################")

    newInput = Input(batch_shape=inputShape)
    newOutputs = face_classification_model(newInput)
    newModel = Model(newInput, newOutputs)
    #print("#######################################################################################")
    #newModel.summary()
    #print("#######################################################################################")
    fileName, fileExtension = os.path.splitext(modelFileName)
    modifiedModelFilePath = fileName + ".h5"
    newModel.save(modifiedModelFilePath)

def main(model_fname,inputShape):
    #model_fname = './model/mask_recognition_v4_updated.hdf5'
    #inputShape = (1,64,64,1)
    modifiedInputLayer(model_fname,inputShape)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shape', nargs='+',required=True, default=[1, 64,64, 1], help='Model input shape.')
    parser.add_argument('--model_file', type=str, default='/model/mask_recognition_v4_updated.hdf5', help='Model input shape.')

    args = parser.parse_args()
    if (args.model_file == '/model/mask_recognition_v4_updated.hdf5'):
        modelFilePath = os.getcwd() + args.model_file
    else:
        modelFilePath = args.model_file

    print("#######################################################################################")
    print(modelFilePath)
    print(args.input_shape)
    print("#######################################################################################")
    main(modelFilePath,args.input_shape)