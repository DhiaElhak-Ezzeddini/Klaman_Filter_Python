import numpy as np
import cv2

lo=np.array([80, 50, 50])
hi=np.array([100, 255, 255])

def detect_inrange(image, surface):
    points=[]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image=cv2.blur(image, (5, 5))
    mask=cv2.inRange(image, lo, hi)
    mask=cv2.erode(mask, None, iterations=2)
    mask=cv2.dilate(mask, None, iterations=2)
    elements=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elements=sorted(elements, key=lambda x:cv2.contourArea(x), reverse=True)
    for element in elements:
        if cv2.contourArea(element)>surface:
            ((x, y), rayon)=cv2.minEnclosingCircle(element)
            points.append(np.array([int(x), int(y)]))
        else:
            break

    return points, mask

def detect_visage(image,use_profile: bool=False):
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    points=[]
    
    
    frontal_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")    
    
    frontal_faces=frontal_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    profile_faces=profile_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
    flipped = cv2.flip(gray,1)
    right_face = profile_cascade.detectMultiScale(flipped, scaleFactor=1.2, minNeighbors=3)
     # Convert right-profile coordinates
    for (x, y, w, h) in right_face:
        x = gray.shape[1] - x - w
        profile_faces = np.append(right_face, [[x, y, w, h]], axis=0)
    # Combine frontal + profile faces safely
    if use_profile :
        if len(frontal_faces) > 0 and len(profile_faces) > 0:
            faces = np.concatenate((frontal_faces, profile_faces), axis=0)
        elif len(frontal_faces) > 0:
            faces = frontal_faces
        elif len(profile_faces) > 0:
            faces = profile_faces
        else:
            faces = []
    else : 
        if len(frontal_faces) > 0:
            faces = frontal_faces
        else :
            faces = []
    
    for(x, y, w, h) in faces:
        points.append(np.array([int(x+w/2), int(y+h/2)]))

    return points, None