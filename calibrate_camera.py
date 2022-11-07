#pip install opencv-contrib-python
import cv2
import sys
import pickle
import glob
from time import time, sleep
import numpy as np
import json
from datetime import datetime
import os

#Classe de erro
class Calibration_Error(Exception):
    pass

if len(sys.argv) > 1:
    camera_id = int(sys.argv[1])
else:
    camera_id = int(input("Camera id: "))
print('camera:', camera_id)

#data das fotos da calibração YYYY-MM-DD
data = '2022-11-01' #datetime.today() 
intrinsec_path = 'calibration_img\\intrinsic\\data_'+str(data)+'_camera'+str(camera_id)+'_'
extrinsec_path = 'calibration_img\\extrinsic\\data_'+str(data)+'_camera'+str(camera_id)+'_'

CHARUCOBOARD_ROWCOUNT = 4
CHARUCOBOARD_COLCOUNT = 3
squareLength = 0.190
markerLength = 0.148
image_h = 728
image_w = 1288
markerSize = 0.4 # Arucao

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
for image_path in (os.listdir('./calibration_img/intrinsic/')):
    try:
        img = cv2.imread('calibration_img/intrinsic/'+image_path,0)
    except:
        continue
    try:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image=img,
            dictionary=ARUCO_DICT)
    except:
        print(image_path,'error')
        continue
    # Outline the aruco markers found in our query image
    img = cv2.aruco.drawDetectedMarkers(
        image=img,
        corners=corners)

    # Get charuco corners and ids from detected aruco markers
    if ids is None:
        continue

    response, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=img,
        board=CHARUCO_BOARD)

    # If a Charuco board was found, let's collect image/corner points
    # Requiring at least 20 squares

    #Must configure the threshold for response, i got 5 at LabTef in 2022-11-01
    print(response)
    if response > 5:
        # Add these corners and ids to our calibration arrays
        corners_all.append(charuco_corners)
        ids_all.append(charuco_ids)

        # # Draw the Charuco board we've detected to show our calibrator the board was properly detected
        img = cv2.aruco.drawDetectedCornersCharuco(
            image=img,
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids)

        # # If our image size is unknown, set it now
        if not image_size:
            image_size = img.shape[::-1]

        cv2.imshow('Charuco board', img)
        cv2.waitKey(1)

    # sleep(1)

  # Destroy any open CV windows
cv2.destroyAllWindows()

print("num_images: {}".format(len(corners_all)))

# Make sure we were able to calibrate on at least one charucoboard by checking
# if we ever determined the image size
if not image_size:
    # Calibration failed because we didn't see any charucoboards of the PatternSize used
    print("Calibration was unsuccessful. We couldn't detect charucoboards in any of the images supplied. Try changing the patternSize passed into Charucoboard_create(), or try different pictures of charucoboards.")
    # Exit for failure

# Now that we've seen all of our images, perform the camera calibration
# based on the set of points we've discovered
else:
    print("Calibrando...")
    calibration_error, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=corners_all,
        charucoIds=ids_all,
        board=CHARUCO_BOARD,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None)

# print(cameraMatrix.reshape(-1).tolist())
# Print matrix and distortion coefficient to the console
print(f"cameraMatrix = {cameraMatrix}")
print(f"distCoeffs = {distCoeffs}")
print(f"calibration = {calibration_error}")


for image_path in (os.listdir('./calibration_img/extrinsic/')):

    try:
        img = cv2.imread(extrinsic_path+image_path,0)
    except:
        continue

    image_h, image_w = img.shape[:2]

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(img, ARUCO_DICT)
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, markerSize, cameraMatrix, distCoeffs)

    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    #7?
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
			#Poderia alterar para rows e cols serem keys
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


print(json.dumps(calibration_parameters))
