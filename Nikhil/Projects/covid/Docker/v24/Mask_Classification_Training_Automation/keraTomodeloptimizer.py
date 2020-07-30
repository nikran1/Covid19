import os
import argparse

def keraTomodeloptimizer(model, input_shape, data_type):

    modelFileDir = os.getcwd() + '/model/mask_recognition_v4_updated.hdf5'

    input_layer_shape = str(1) + " " +str(input_shape[0]) + " " +str(input_shape[1]) + " " +str(input_shape[2]) + " "
    # input_shape = str(args.input_shape[0]) + " " +str(args.input_shape[1]) + " " +str(args.input_shape[2]) + " " +str(args.input_shape[3]) + " "
    os.system("python3 inputLayerReplace.py --input_shape "+ input_layer_shape +" --model_file "+'/model/mask_recognition_v4_updated.hdf5')


    fileName, fileExtension = os.path.splitext(modelFileDir)
    filePath, _ = os.path.split(modelFileDir)
    modifiedModelFilePath = fileName + ".h5"
    os.system("python3 kerasToOnnx.py --model_file "+ modifiedModelFilePath)

    onnxModelFilePath = fileName +'.onnx'
    #change current working directory as openvino for initialize
    os.chdir("/opt/intel/openvino_2020.3.194/bin")
    #os.chdir("C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/bin")
    os.system("/opt/intel/openvino/bin/setupvars.sh")
    #os.system("setupvars.bat")
    print()
    print(os.getcwd())
    os.chdir("/opt/intel/openvino_2020.3.194/deployment_tools/model_optimizer")
    #os.chdir("C:/Program Files (x86)/IntelSWTools/openvino_2019.3.379/deployment_tools/model_optimizer")
    print()
    print(os.getcwd())
    mo_input_shape = "[" + str(1) + "," + str(input_shape[0]) + "," + str(input_shape[1]) + "," + str(input_shape[2]) +"]"
    os.system("python3 mo.py --input_model "+ onnxModelFilePath + " --output_dir " + filePath + " --input_shape "+ mo_input_shape +" --data_type "+ str(data_type))
