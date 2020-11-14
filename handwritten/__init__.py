import logging
import azure.functions as func
import numpy as np
from PIL import Image
#import io
from . import preprocessing as prep 
from time import time
###1106
from .utils.codec import CTCCodec
import grpc
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
###
_HOST = 'ovaasbackservertest.japaneast.cloudapp.azure.com'
_PORT = '9002'

def main(req: func.HttpRequest,context: func.Contex) -> func.HttpResponse:
    _NAME = 'image'
    
    event_id = context.invocation_id
    logging.info(f"Python humanpose function start process.\nID:{event_id}\nback server host:{_HOST}:{_PORT}")
   
    method = req.method
    url    = req.url
    params = req.params

    if method != 'POST':
        logging.warning(f'ID:{event_id},the method was {files.content_type}.refused.')
        return func.HttpResponse(f'only accept POST method',status_code=400)

    try:
        files = req.files[_NAME]
        if files:
            if files.content_type != 'image/jpeg':
                logging.warning(f'ID:{event_id},the file type was {files.content_type}.refused.')
                return func.HttpResponse(f'only accept jpeg images',status_code=400)

            #get japanese_char_list by char_list_path
            characters = prep.get_characters("data/kondate_nakayosi_char_list.txt")
            codec = CTCCodec(characters)

            # pre processing
            img = files.read()
            ##img = prep.to_pil_image(img) 
            input_batch_size, input_channel, input_height, input_width= (1,1,96,2000)
            input_image = prep.preprocess_input(img, height=input_height, width=input_width)[None,:,:,:]

            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'handwritten-japanese-recognition'
            request.inputs["data"].CopyFrom(make_tensor_proto(input_image, shape=input_image.shape))

            #send to infer model by grpc
            start = time()
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            output = stub.Predict(request, 10.0)


            
            timecost = time()-start_time
            logging.warning(f"Inference complete,Takes{timecost}")


            result = codec.decode(output["output"])
            return func.HttpResponse(f"Did you wirte {result}!! This HTTP triggered function executed successfully.")

        else:
            logging.warning(f'ID:{event_id},Failed to get image,down.')
            return func.HttpResponse(f'no image files', status_code=400)

    except grpc.RpcError as e:
        status_code = e.code()
        if "DEADLINE_EXCEEDED" in status_code.name:
            logging.error(e)
            return func.HttpResponse(f'the grpc request timeout', status_code=408)
        else:
            logging.error(f"grpcError:{e}")
            return func.HttpResponse(f'Failed to get grpcResponse', status_code=500)

    except Exception as e:
        logging.error(f"Error:{e}\n\
                        url:{url}\n\
                        method:{method}\n\
                        params:{params}")
        return func.HttpResponse(f'Service Error.check the log.',status_code=500)