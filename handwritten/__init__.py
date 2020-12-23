# coding: utf-8
import sys
import os
import json
import logging

import azure.functions as func
import numpy as np
from PIL import Image
from . import preprocessing as prep
from time import time
###1106
from .codec import CTCCodec 
import grpc
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import subprocess
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import chardet

# TODO
# make a exception processing for if result == Null

#_HOST = 'ovaasbackservertest.japaneast.cloudapp.azure.com'
_HOST = os.environ["VM_IPADDRESS"]
_PORT = '10003'
def main(req: func.HttpRequest,context: func.Context) -> func.HttpResponse:

    _NAME = 'image'
    
    event_id = context.invocation_id

    logging.info(f"Python handwritten function start process.\nID:{event_id}\nback server host:{_HOST}:{_PORT}")
   
   

    try:
        method = req.method
        url    = req.url
        params = req.params
        files = req.files[_NAME]

        if method != 'POST':
            logging.warning(f'ID:{event_id},the method was {files.content_type}.refused.')
            return func.HttpResponse(f'only accept POST method',status_code=400)
        
        if files:
            if files.content_type != 'image/jpeg':
                logging.warning(f'ID:{event_id},the file type was {files.content_type}.refused.')
                return func.HttpResponse(f'only accept jpeg images',status_code=400)

            #get japanese_char_list by char_list_path
            # logging.info(os.getcwd())
            CHARSET_PATH = "./handwritten/kondate_nakayosi_char_list.txt"
            characters = prep.get_characters(CHARSET_PATH)
            codec = CTCCodec(characters)
            logging.info(f'Codec Success')
            # pre processing
            img_bin= files.read()
            img = prep.to_pil_image(img_bin)
            if np.array(img).ndim == 3:
                img = np.array(img)[:, :, 0]
            else:
                img = np.array(img)
            logging.info(f'img.shape{img.shape}')
            #logging.warning(f'img.shape2{np.array(img)[:, :, 0].shape}')
            #FIXED the width is too long
            input_batch_size, input_channel, input_height, input_width= (1,1,96,2000)
            input_image = prep.preprocess_input(img, height=input_height, width=input_width)[None,:,:,:]
            #input_image = prep.preprocess_input(np.array(img)[:, :, 0], height=input_height, width=input_width)[None,:,:,:]
            logging.info(f'Input_Image Success')

            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'handwritten-japanese-recognition'
            request.inputs["actual_input"].CopyFrom(make_tensor_proto(input_image, shape=input_image.shape))
            logging.info(f'Requse Detail  Success')
            
            #send to infer model by grpc
            
            start = time()
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            output = stub.Predict(request,timeout = 10.0)
            logging.info(f'Grpc Success{output}')
            result = make_ndarray(output.outputs["output"])


            
            timecost = time()-start
            logging.warning(f"Inference complete,Takes{timecost}")

            text = codec.decode(result)
            logging.info(f"TextLength{len(text[0])}")
            logging.info(f"TextType{type(text[0])}")
            #Error: Words are garbled
#             logging.info(chardet.detect(text[0].encode()))
            #text[0] = text[0].encode().decode('utf-8')
#             text = text[0].encode().decode('utf-8')
            if len(text[0]) == 0:
                return func.HttpResponse(f'AI model could not understand your handwriting', status_code=404)
            
            text = text[0]
    
            #logging.warning(f'Azure Function has{subprocess.call('echo $LANG', shell=True)}')
            #FIXIT just response result and status code
            logging.info(f'Text Content{text}')
   
            '''
            #text = text[0].encode("utf-8")
            file_name = "word.txt"
            fileobj = open(file, "w", encoding = "utf_8")
            fileobj.write(text[0])
            fileobj.close()
             
            
            
            #Changing string into jpeg
            ttfontname = "japanese_font.ttc"
            logging.warning(f'fontfile success')
            fontsize = 45
            logging.warning(f'string detail success')
            canvasSize    = ((len(text)+3)*fontsize,90)
            backgroundRGB = (255, 255, 255)
            textRGB       = (0, 0, 0)
            logging.warning(f'string color success')
            word_img  = PIL.Image.new('RGB', canvasSize, backgroundRGB)
            draw = PIL.ImageDraw.Draw(word_img)
            logging.warning(f'draw success')

            font = PIL.ImageFont.truetype(ttfontname, fontsize)
            textWidth, textHeight = draw.textsize(text,font=font)
            textTopLeft = (canvasSize[0]//6, canvasSize[1]//2-textHeight//2)
            draw.text(textTopLeft, text, fill=textRGB, font=font)
            logging.warning(f'text success')
            imgbytes = word_img.tobytes()
            '''

            
#             MIMETYPE =  'image/jpeg'
            #return func.HttpResponse(body=imgbytes, status_code=200,mimetype=MIMETYPE,charset='utf-8')
            return func.HttpResponse(json.dumps({
                'count': len(text),
                'timecost': timecost,
                'text': text
            }))         
            
            #return func.HttpResponse(f'{text[0]}')


        else:
            logging.warning(f'ID:{event_id},Failed to get file,down.')
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
