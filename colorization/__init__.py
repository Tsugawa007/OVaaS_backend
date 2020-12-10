import logging
import azure.functions as func

from time import time
from PIL import Image
from . import preprocessing as prep
import cv2


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _HOST = prep.__get_config__('COLORIZATION', 'host')
    _PORT = prep.__get_config__('COLORIZATION', 'port')
    _NAME = prep.__get_config__('COLORIZATION', 'name')
    _MODEL_NAME = prep.__get_config__('COLORIZATION', 'model_name')

    event_id = context.invocation_id
    logging.info(f"Python colorization function start process.\nID:{event_id}\nback server host:{_HOST}:{_PORT}")

    method = req.method
    url = req.url
    files = req.files[_NAME]
    if method != 'POST':
        logging.warning(f'ID:{event_id},the method was {files.content_type}.refused.')
        return func.HttpResponse(f'only accept POST method', status_code=400)
    if not files:
        logging.warning(f'ID:{event_id},Failed to get image,down.')
        return func.HttpResponse(f'no image files', status_code=400)
    
    start_time = time()

    # pre processing
    input_source = files.read()  # get image_bin form request
    try:
        colorization = prep.RemoteColorization(_HOST, _PORT, _MODEL_NAME)
        original_frame = cv2.imread(input_source)
        img_bgr_out = colorization.infer(original_frame)
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

    logging.info(f"Successfully.Create output image")
    final_image = prep.create_output_image(original_frame, img_bgr_out)
    mimetype = 'image/jpeg'
    img_output_bytes = Image.fromarray(final_image)

    return func.HttpResponse(body=img_output_bytes, status_code=200, mimetype=mimetype, charset='utf-8')
