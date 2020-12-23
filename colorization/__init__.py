import logging
import azure.functions as func

from time import time
from . import preprocessing as prep
import cv2


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _HOST = os.environ.get("VM_IPADDRESS")
    _PORT = os.environ.get("COLORIZATION_PORT")
    #_HOST = prep.__get_config__('COLORIZATION', 'host')
    #_PORT = prep.__get_config__('COLORIZATION', 'port')
    _NAME = prep.__get_config__('COLORIZATION', 'name')
    _MODEL_NAME = prep.__get_config__('COLORIZATION', 'model_name')

    event_id = context.invocation_id
    logging.info(f"Python colorization function start process.\nID:{event_id}\nback server host:{_HOST}:{_PORT}")

    method = req.method
    url = req.url
    # files = req.files[_NAME]
    files = req.files['image']
    logging.info(f"files is {files}")
    if method != 'POST':
        logging.warning(f'ID:{event_id},the method was {files.content_type}.refused.')
        return func.HttpResponse(f'only accept POST method', status_code=400)
    if not files:
        logging.warning(f'ID:{event_id},Failed to get image,down.')
        return func.HttpResponse(f'no image files', status_code=400)
    
    start_time = time()

    # pre processing
    input_image = prep.create_input_image(files)  # get image form request
    logging.info(f'Input_Image Success.')
    # original_frame = cv2.imread(input_image)
    try:
        img_bgr_out = prep.RemoteColorization(_HOST, _PORT, _MODEL_NAME).infer(input_image) 
        #colorization = prep.RemoteColorization(_HOST, _PORT, _MODEL_NAME)
       
        logging.info(f"outpre success!")
        #img_bgr_out=colorization.infer(input_image)
    except Exception as e:
        if 'StatusCode.DEADLINE_EXCEEDED' in str(e):
            logging.error(e)
            return func.HttpResponse(f'The gRPC request time out', status_code=408)
        else:
            logging.error(f"Error:{e}\n\
                            url:{url}\n\
                            method:{method}\n")
            return func.HttpResponse(f'Service Error.check the log.', status_code=500)

    time_cost = time() - start_time

    logging.info(f"Inference complete,Takes{time_cost}")

    logging.info(f"Successfully.img_bgr_out is {img_bgr_out}.\nCreate output image.")
    #final_image = prep.create_output_image(img_bgr_out, img_bgr_out)
    final_image = prep.create_output_image(input_image, img_bgr_out)
    logging.info(f"Successfully.final_image is {final_image}.")
    mimetype = 'image/jpeg'
    
    # img_output_bytes = Image.fromarray(final_image)
    img_output_bytes = prep.cv2ImgToBytes(final_image)
    logging.info(f"Successfully.img_output_bytes is {img_output_bytes}.")
    return func.HttpResponse(body=img_output_bytes, status_code=200, mimetype=mimetype, charset='utf-8')
