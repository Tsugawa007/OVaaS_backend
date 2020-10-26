import logging

import azure.functions as func
import numpy as np
from PIL import Image
import io
from . import preprocessing as prep 
from . import postprocessing as posp
from ..grpc_client.grpc_humanpose_client import run as client
from time import time

'''
Post Analysis:

    header: content-type:multipart/form-data
    body: 	
    Content-Disposition: form-data; name="image"; filename="xxx.jpeg or xxx.png"
        Content-Type: image/jpeg or image/png
        
        image binary file
    *****************************************************
    file size: <500kb
    file analysis: .jpg/.jpeg only


'''


def main(req: func.HttpRequest) -> func.HttpResponse:
    _NAME = 'image'

    logging.info("Python HTTP trigger function processed a request")
    # header = req.headers.items()
    # for i in header:
    #     print(i)
    method = req.method
    url    = req.url
    params = req.params
    try:
        files = req.files[_NAME]
        if files:
            #pre processing
            img_bin = files.read()      #get image_bin form request
            img = prep.to_pil_image(img_bin)
            img = prep.resize(img) # w,h = 456,256        
            img_np = np.array(img)
            img_np = prep.transpose(img_np) #hwc > bchw [1,3,256,456]

            # send to model server by grpc
            start = time()
            people = client(img_np)
            
            if not people:
                timecost = time()-start
                logging.warning(f"Inference complete,But no person detected,Takes{timecost}")
                return func.HttpResponse(f'No person detected',status_code=200)
            
            timecost = time()-start
            logging.info(f"Inference complete,Takes{timecost}")
            #post processing
            img_fin = posp.post_processing(img_np,people).res
            MIMEType = 'image/jpeg'
            return func.HttpResponse(body=img_fin,status_code=200,mimetype=MIMEType)

        else:
            return func.HttpResponse(f'no image files',status_code=400)
    except Exception as e:
        logging.error(f"Error:{e}\n\
                        url:{url}\n\
                        method:{method}\n\
                        params:{params}")
        return func.HttpResponse(f'Service Error.check the log.',status_code=500)



    # img.show() #for local debug


    
    