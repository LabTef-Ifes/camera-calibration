# pip install opencv-contrib-python
import cv2
import sys
import pickle
import glob
from time import time, sleep
import numpy as np
import json
from datetime import datetime
import os

# Classe de erro
class Calibration_Error(Exception):
    pass


if len(sys.argv) > 1:
    camera_id = int(sys.argv[1])
else:
    camera_id = int(input("Camera id: "))
print('camera:', camera_id)

# data das fotos da calibração YYYY-MM-DD
data = '2022-11-07'  # datetime.today()
intrinsec_path = 'calibration_img\\intrinsic\\'

#\data_' + str(data)+'_camera'+str(camera_id)+'_'
extrinsec_path = 'calibration_img\\extrinsic\\'
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


# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = []  # Corners discovered in all images processed
ids_all = []  # Aruco ids corresponding to corners discovered
image_size = None  # Determined at runtime


# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
#camera_intrinsec = cv2.VideoCapture("./output/intrinsec_cam{}.avi".format(camera_id))
#camera_extrinsec = cv2.VideoCapture("./output/extrinsec_cam{}.avi".format(camera_id))
# Loop through images glob'ed
#crop = np.array((153,478,228,558))

for image_path in (os.listdir('./calibration_img/extrinsic/')):

    try:
        if "camera"+str(camera_id) in image_path:
            img = cv2.imread('calibration_img/extrinsic/'+image_path, 0)

        else:
            continue
    except:
        continue

    image_h, image_w = img.shape[:2]

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(img, ARUCO_DICT)
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, markerSize, cameraMatrix, distCoeffs)

    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # 7?
    marker_id = np.where(ids.reshape(-1) == 7)
    rvec = rvec[marker_id].reshape((1, -1))
    print(rvec.shape, ids, rvec.shape)
    rotation, _ = cv2.Rodrigues(rvec)
    translation = tvec[marker_id].reshape((3, 1))
    print("//", rotation.shape, translation.shape,
          np.array([[0, 0, 0, 1]]).shape)
    extrinsecs = cv2.hconcat([rotation, translation])
    print(extrinsecs.shape)
    last_colum = np.ones((1, 4))
    #last_colum[:, -1] = 1
    extrinsecs = cv2.vconcat([extrinsecs, last_colum])
    print(extrinsecs.shape, cameraMatrix.shape, distCoeffs.shape)
    break

calibration_parameters = {
    "error": calibration_error,
    "resolution": {
        "height": image_h,
        "width": image_w,
    },
    "id": "{}".format(camera_id),
    "extrinsic": {
        "to": "{}".format(camera_id),
        "tf": {
          "shape": {
              "dims": [
                  {
                      "name": "rows",
                      "size": CHARUCOBOARD_ROWCOUNT
                  },
                  {
                      "name": "cols",
                      "size": CHARUCOBOARD_COLCOUNT
                  }
              ]
          },
            "type": "DOUBLE_TYPE",
            "doubles": extrinsecs.reshape(-1).tolist()
        },
        "from": "1000"
    },
    "calibratedAt": data,
    "intrinsic": {
        "shape": {
            # Poderia alterar para rows e cols serem keys
            "dims": [
                {
                    "name": "rows",
                    "size": cameraMatrix.shape[0]
                },
                {
                    "name": "cols",
                    "size": cameraMatrix.shape[1]
                }
            ]
        },
        "type": "DOUBLE_TYPE",
        "doubles": cameraMatrix.reshape(-1).tolist()
    },
    "distortion": {
        "shape": {
            "dims": [
                {
                    "name": "rows",
                    "size": distCoeffs.shape[0]
                },
                {
                    "name": "cols",
                    "size": distCoeffs.shape[1]
                }
            ]
        },
        "type": "DOUBLE_TYPE",
        "doubles": distCoeffs.reshape(-1).tolist()
    }
}

with open('params_camera{}.json'.format(camera_id), 'w') as f:
    json.dump(calibration_parameters, f, indent=2)


print(json.dumps(calibration_parameters), indent=2)
