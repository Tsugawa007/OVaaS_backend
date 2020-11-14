import numpy as np
import cv2
from PIL import Image
import base64
import io
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2


def to_pil_image(img_bin):
    _decoded = io.BytesIO(img_bin)
    return Image.open(_decoded)



def __get_input_name_and_shape__():
    metadata_field = "signature_def"
    request = get_model_metadata_pb2.GetModelMetadataRequest()
    request.model_spec.name = 'colorization'
    request.metadata_field.append(metadata_field)

    result = self.stub.GetModelMetadata(request, 10.0) # result includes a dictionary with all model outputs
    input_metadata, output_metadata = __get_input_and_output_meta_data__(result)
    input_blob = next(iter(input_metadata.keys()))
    output_blob = next(iter(output_metadata.keys()))
    return input_blob, input_metadata[input_blob]['shape'], output_blob, output_metadata[output_blob]['shape']
    



def __get_input_and_output_meta_data__(response):
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

    return input_blobs_keys, output_blobs_keys





def __preprocess_input__(original_frame,input_width,input_height,input_batchsize,input_channel):
    if original_frame.shape[2] > 1:
        frame = cv2.cvtColor(cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    else:
        frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)

    img_rgb = frame.astype(np.float32) / 255
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    img_l_rs = cv2.resize(img_lab.copy(), (input_width, input_height))[:, :, 0]
    input_image = img_l_rs.reshape(input_batchsize, input_channel, input_height, input_width).astype(np.float32)
        
    return input_image





