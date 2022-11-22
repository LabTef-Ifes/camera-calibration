from is_wire.core import Channel,Subscription,Message
from is_msgs.image_pb2 import Image
import numpy as np
import cv2
import json
import time
from datetime import date, datetime

camera_id = 1 # selecionar camera
h = 728
w = 1288
today = date.today()

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

    print('---RUNNING THE CAMERA CLIENT---')
    CHARUCOBOARD_ROWCOUNT = 4
    CHARUCOBOARD_COLCOUNT = 3
    squareLength = 0.20
    markerLength = 0.16
    image_h = 728
    image_w = 1288
    markerSize = 0.649  # Arucao
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    # Create constants to be passed into OpenCV and Aruco methods
    CHARUCO_BOARD = cv2.aruco.CharucoBoard_create(
    squaresX=CHARUCOBOARD_COLCOUNT,
    squaresY=CHARUCOBOARD_ROWCOUNT,
    squareLength=squareLength,
    markerLength=markerLength,
    dictionary=ARUCO_DICT)

    broker_uri = "amqp://guest:guest@localhost:5672"
    channel = Channel(broker_uri)
    subscription = Subscription(channel=channel)
    subscription.subscribe(topic='CameraGateway.{}.Frame'.format(camera_id))

    print(channel)

    window = 'Blackfly S Camera'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, h, w)
    n=0
    while True:
        msg = channel.consume()  
        im = msg.unpack(Image)
        frame = to_np(im)
        try:
            corners, ids, _ = cv2.aruco.detectMarkers(
                image=frame,
                dictionary=ARUCO_DICT)
        except:
            print('error')
            continue
        # Outline the aruco markers found in our query image
        img = cv2.aruco.drawDetectedMarkers(
            image=frame,
            corners=corners)

        # Get charuco corners and ids from detected aruco markers
        if ids is None:
            continue

        response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=img,
            board=CHARUCO_BOARD)
        print(response)



        corners, ids, rejected = cv2.aruco.detectMarkers(img, ARUCO_DICT)
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, markerSize, cameraMatrix, distCoeffs)

        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        cv2.imshow(window, frame)
        key = cv2.waitKey(1)
        
        # Stops the recording
        if key == ord('q'):
            break
        
        # Saves the frame on pressing 's'
        if key == ord('s'):
            n+=1
            cv2.imwrite(f"calibration_img/intrisec/cam{camera_id}/data_{today}_camera{camera_id}_{n:0>3}.png", frame)
            print(f'imagem {n} capturada')

	
	
