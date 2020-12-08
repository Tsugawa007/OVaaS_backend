import numpy as np
import cv2
from PIL import Image
import io
from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2
import configparser
import grpc
# ファイルの存在チェック用モジュール
import os
import errno
import logging

class PreProcessing:

    def __init__(self, grpc_address='localhost', grpc_port=9000, model_name='colorization', model_version=None):
        logging.info(f'start init') # Delete this line when the test is complete
        # Settings for accessing model server
        self.grpc_address = grpc_address
        self.grpc_port = grpc_port
        self.model_name = model_name
        self.model_version = model_version
        logging.info(f'set self: grpc_address is {grpc_address}\n grpc_port is {grpc_port}\nmodel_name is {model_name}\n model_version is {model_version}') # Delete this line when the test is complete
        channel = grpc.insecure_channel("{}:{}".format(self.grpc_address, self.grpc_port))
        logging.info(f'set grpc channel Success.channel is {channel}') # Delete this line when the test is complete
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # Get input shape info from Model Server
        self.input_name, input_shape, self.output_name, output_shape = self.__get_input_name_and_shape__()
        self.input_batchsize = input_shape[0]
        self.input_channel = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

        # Setup coeffs
        # coeffs = "public/colorization-v2/colorization-v2.npy"
        coeffs = "colorization-v2.npy"
        self.color_coeff = np.load(coeffs).astype(np.float32)
        logging.info(f'color_coeff Success') # Delete this line when the test is complete
        assert self.color_coeff.shape == (313, 2), "Current shape of color coefficients does not match required shape"
        logging.info(f'init Success') # Delete this line when the test is complete

    def __to_pil_image__(self, img_bin):
        _decoded = io.BytesIO(img_bin)
        return Image.open(_decoded)

    def __get_input_name_and_shape__(self):
        logging.info(f'start  get_input_name_and_shape') # Delete this line when the test is complete
        metadata_field = "signature_def"
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = self.model_name
        if self.model_version is not None:
            request.model_spec.version.value = self.model_version
        request.metadata_field.append(metadata_field)

        result = self.stub.GetModelMetadata(request, 10.0)  # result includes a dictionary with all model outputs
        input_metadata, output_metadata = self.__get_input_and_output_meta_data__(result)
        input_blob = next(iter(input_metadata.keys()))
        output_blob = next(iter(output_metadata.keys()))
        logging.info(f'color_coeff Success') # Delete this line when the test is complete
        return input_blob, input_metadata[input_blob]['shape'], output_blob, output_metadata[output_blob]['shape']

    def __get_input_and_output_meta_data__(self, response):
        logging.info(f'start  get_input_and_output_meta_data') # Delete this line when the test is complete
        signature_def = response.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)
        serving_default = signature_map.ListFields()[0][1]['serving_default']
        serving_inputs = serving_default.inputs
        input_blobs_keys = {key: {} for key in serving_inputs.keys()}
        tensor_shape = {key: serving_inputs[key].tensor_shape
                        for key in serving_inputs.keys()}
        for input_blob in input_blobs_keys:
            inputs_shape = [d.size for d in tensor_shape[input_blob].dim]
            tensor_dtype = serving_inputs[input_blob].dtype
            input_blobs_keys[input_blob].update({'shape': inputs_shape})
            input_blobs_keys[input_blob].update({'dtype': tensor_dtype})

        serving_outputs = serving_default.outputs
        output_blobs_keys = {key: {} for key in serving_outputs.keys()}
        tensor_shape = {key: serving_outputs[key].tensor_shape
                        for key in serving_outputs.keys()}
        for output_blob in output_blobs_keys:
            outputs_shape = [d.size for d in tensor_shape[output_blob].dim]
            tensor_dtype = serving_outputs[output_blob].dtype
            output_blobs_keys[output_blob].update({'shape': outputs_shape})
            output_blobs_keys[output_blob].update({'dtype': tensor_dtype})
        logging.info(f'get_input_and_output_meta_data Success') # Delete this line when the test is complete
        return input_blobs_keys, output_blobs_keys

    def __preprocess_input__(self, original_frame):
        logging.info(f'start  preprocess_input') # Delete this line when the test is complete
        if original_frame.shape[2] > 1:
            frame = cv2.cvtColor(cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        img_l_rs = cv2.resize(img_lab.copy(), (self.input_width, self.input_height))[:, :, 0]
        logging.info(f'preprocess_input Success') # Delete this line when the test is complete
        return img_lab, img_l_rs


def __get_config__(section, key):
    # iniファイルの読み込み
    config_ini = configparser.ConfigParser()
    config_ini_path = os.path.split(os.path.realpath(__file__))[0] + '/config.ini'

    # 指定したiniファイルが存在しない場合、エラー発生
    if not os.path.exists(config_ini_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_ini_path)

    config_ini.read(config_ini_path, encoding='utf-8')
    return config_ini.get(section, key)
