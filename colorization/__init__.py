import logging
import azure.functions as func
import numpy as np

from time import time

from . import preprocessing as prep
from . import postprocessing as posp

import grpc
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2

import cv2

# FIXIT CAN NOT RUN!.
# multiendpoint 
# maybe should rewrite the azure configure file
_HOST = prep.__get_config__('COLORIZATION', 'host')
_PORT = prep.__get_config__('COLORIZATION', 'port')


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _NAME = prep.__get_config__('COLORIZATION', 'name')
    _MODEL_NAME = prep.__get_config__('COLORIZATION', 'model_name')

    event_id = context.invocation_id
    logging.info(f"Python {_MODEL_NAME} function start process.\nID:{event_id}\nback server host:{_HOST}:{_PORT}")

    try:
        method = req.method
        url = req.url
        header = req.headers
        files = req.files[_NAME]

        if method != 'POST':
            logging.warning(f'ID:{event_id},the method was {files.content_type}.refused.')
            return func.HttpResponse(f'only accept POST method', status_code=400)

        if files:
            if files.content_type != 'image/jpeg':
                logging.warning(f'ID:{event_id},the file type was {files.content_type}.refused.')
                return func.HttpResponse(f'only accept jpeg images', status_code=400)

            # pre processing
            img_bin = files.read()  # get image_bin form request
            pre_process = prep.PreProcessing(model_name=_MODEL_NAME)
            original_frame = pre_process.__to_pil_image__(img_bin)

            # Read and pre-process input image (NOTE: one image only)
            img_lab, img_l_rs = pre_process.__preprocess_input__(original_frame)
            input_image = img_l_rs.reshape(pre_process.input_batchsize, pre_process.input_channel,
                                           pre_process.input_height,
                                           pre_process.input_width).astype(np.float32)

            # res = self.exec_net.infer(inputs={self.input_blob: [img_l_rs]})
            # Model ServerにgRPCでアクセスしてモデルをコール
            request = predict_pb2.PredictRequest()
            request.model_spec.name = pre_process.model_name
            request.inputs[pre_process.input_name].CopyFrom(make_tensor_proto(input_image, shape=(input_image.shape)))
            start_time = time()
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            result = stub.Predict(request, 10.0)  # result includes a dictionary with all model outputs
            res = make_ndarray(result.outputs[pre_process.output_name])

            update_res = (res * pre_process.color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)

            out = update_res.transpose((1, 2, 0))
            (h_orig, w_orig) = original_frame.shape[:2]
            out = cv2.resize(out, (w_orig, h_orig))
            img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
            img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

            time_cost = time() - start_time
            logging.info(f"Inference complete,Takes{time_cost}")

            # post processing
            final_image = posp.create_output_image(original_frame, img_bgr_out)
            img_bytes = cv2.imencode('.jpg', final_image)[1].tobytes()
            mimetype = 'image/jpeg'

            return func.HttpResponse(body=img_bytes, status_code=200, mimetype=mimetype, charset='utf-8')

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
