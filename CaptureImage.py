from is_wire.core import Channel, Subscription, Message
from is_msgs.image_pb2 import Image
import numpy as np
import cv2
import json
import time
import glob
import datetime

# alterar o ID da camera e o diretório

camera_id = 2
h = 720
w = 1280
today = datetime.datetime.today()

def to_np(input_image):
    if isinstance(input_image, np.ndarray):
        return input_image
    elif isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        output_image = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    else:
        output_image = np.array([], dtype=np.uint8)
    return output_image


if __name__ == '__main__':
    print('---RUNNING EXAMPLE DEMO OF THE CAMERA CLIENT---')
    broker_uri = "amqp://10.10.3.188:30000"
    channel = Channel(broker_uri)
    subscription = Subscription(channel=channel, name="Intelbras_Camera")
    subscription.subscribe(topic='CameraGateway.{}.Frame'.format(1))

    window = 'Intelbras Camera'
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", h, w)
    n = 0
    img_array = []
    while True:
        msg = channel.consume()
        im = msg.unpack(Image)
        frame = to_np(im)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            n += 1
            cv2.imwrite(
                f'./calibration_img/intrinsic/data_{today}_camera{camera_id}_{n}.png', frame)
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow("camera", frame)
