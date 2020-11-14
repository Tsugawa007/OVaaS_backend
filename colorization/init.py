import logging
import azure.functions as func
import numpy as np
from PIL import Image
import io
import sys
import os
from time import time

from . import preprocessing as prep
from . import postprocessing as posp


import grpc
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2


import traceback
import cv2


_HOST = 'ovaasbackservertest.japaneast.cloudapp.azure.com'
_PORT = '10002'

def main(req: func.HttpRequest) -> func.HttpResponse:
    _NAME = 'image'
    _MODEL_NAME = "colorization"

    event_id = context.invocation_id
    logging.info(f"Python colorization function start process.\nID:{event_id}\nback server host:{_HOST}:{_PORT}")

    method = req.method
    url = req.url
    header = req.headers



    # Setup coeffs
    coeffs = "public/colorization-v2/colorization-v2.npy"
    color_coeff = np.load(coeffs).astype(np.float32)
    assert color_coeff.shape == (313, 2), "Current shape of color coefficients does not match required shape"




    if method != 'POST':
        logging.warning(f'ID:{event_id},the method was {files.content_type}.refused.')
        return func.HttpResponse(f'only accept POST method',status_code=400)

    try:
        files = req.files[_NAME]
        if files:
            if files.content_type != 'image/jpeg':
                logging.warning(f'ID:{event_id},the file type was {files.content_type}.refused.')
                return func.HttpResponse(f'only accept jpeg images',status_code=400)


            # Get input shape info from Model Server
            input_name, input_shape, output_name, output_shape = prep.__get_input_name_and_shape__()
            input_batchsize = input_shape[0]
            input_channel = input_shape[1]
            input_height = input_shape[2]
            input_width = input_shape[3]

            # preprocessing
            img_bin = files.read()  # get image_bin form request
            original_frame = prep.to_pil_image(img_bin)            
            input_image=prep.__preprocess_input__(original_frame,input_width,input_height,input_batchsize,input_channel)
            

            request = predict_pb2.PredictRequest()
            request.model_spec.name = _MODEL_NAME
            request.inputs[self.input_name].CopyFrom(make_tensor_proto(input_image, shape=(input_image.shape)))
            # send to infer model by grpc
            start_time = time()
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            result = stub.Predict(request, 10.0)

            res = make_ndarray(result.outputs[self.output_name])
            update_res = (res * self.color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)
            out = update_res.transpose((1, 2, 0))
            
            timecost = time()-start_time
            logging.info(f"Inference complete,Takes{timecost}")
            


            # post processing
            img_bgr_out=posp.infer(original_frame,out)
            final_image=posp.create_output_image(original_frame, img_bgr_out)
            imgbytes = cv2.imencode('.jpg',final_image)[1].tobytes()
            MIMETYPE =  'image/jpeg'
            
            return func.HttpResponse(body=imgbytes, status_code=200,mimetype=MIMETYPE,charset='utf-8')


        else:
            logging.warning(f'ID:{event_id},Failed to get image,down.')
            return func.HttpResponse(f'no image files', status_code=400)
    except grpc.RpcError as e:
        status_code = e.code()
        if str(status_code) == u'DEADLINE_EXCEEDED':
            logging.error(e)
            return func.HttpResponse(f'the grpc request timeout', status_code=408)
        else:
            logging.error(f"grpcError:{e}")
            return func.HttpResponse(f'Failed to get grpcResponse', stats_code=500)
    
    except Exception as e:
        logging.error(f"Error:{e}\n\
                        url:{url}\n\
                        method:{method}\n")
        return func.HttpResponse(f'Service Error.check the log.', status_code=500)