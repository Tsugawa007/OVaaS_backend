import cv2
import numpy as np


def create_output_image(original_frame, img_bgr_out):
    (h_orig, w_orig) = original_frame.shape[:2]
    im_show_size = (int(w_orig * (200 / h_orig)), 200)
    original_image = cv2.resize(original_frame, im_show_size)
    colorize_image = (cv2.resize(img_bgr_out, im_show_size) * 255).astype(np.uint8)

    original_image = cv2.putText(original_image, 'Original', (25, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    colorize_image = cv2.putText(colorize_image, 'Colorize', (25, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    ir_image = [cv2.hconcat([original_image, colorize_image])]
    final_image = cv2.vconcat(ir_image)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    return final_image
