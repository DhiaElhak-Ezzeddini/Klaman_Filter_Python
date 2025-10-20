import cv2
from Detector import detect_inrange, detect_visage
from KalmanFilter import KalmanFilter
import numpy as np

VideoCap=cv2.VideoCapture(0)

KF=KalmanFilter(0.1, [0, 0])

while(True):
    ret, frame=VideoCap.read()

    #points, mask=detect_inrange(frame, 800)
    points, mask=detect_visage(frame)

    etat=KF.predict().astype(np.int32)

    cv2.circle(frame, (int(etat[0].item()), int(etat[1].item())), 2, (0, 255, 0), 5)
    cv2.arrowedLine(frame,
                    (etat[0].item(), etat[1].item()), (etat[0].item()+etat[2].item(), etat[1].item()+etat[3].item()),
                    color=(0, 255, 0),
                    thickness=3,
                    tipLength=0.2)
    if (len(points)>0):
        cv2.circle(frame, (points[0][0], points[0][1]), 10, (0, 0, 255), 2)
        KF.update(np.expand_dims(points[0], axis=-1))

    cv2.imshow('image', frame)
    if mask is not None:
        cv2.imshow('mask', mask)

    if cv2.waitKey(1)&0xFF==ord('q'):
        VideoCap.release()
        cv2.destroyAllWindows()
        break