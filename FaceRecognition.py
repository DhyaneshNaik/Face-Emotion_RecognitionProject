
import cv2
import numpy as np
import face_recognition
import PIL.ImageDraw
import PIL.Image
import os
from datetime import datetime

path = "ImageAttendance"
images = []
class_names =[]
myList = os.listdir(path)

#model = model_from_json(open("fer.json","r").read())
#model.load_weights("fer_weights.h5")

for cl in myList:
    curImage = cv2.imread(f"{path}/{cl}")
    images.append(curImage)
    #class_names.append(cl.split(".")[0])
    class_names.append(os.path.splitext(cl)[0])
print(class_names)

def find_encodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markeAttendance(images):
    with open("Attendace.csv","r+") as f:
        dataList = f.readlines()
        nameList = []
        for line in dataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")

encodeListKnown = find_encodings(images)
print("Encoding Done. ",len(encodeListKnown))


cap = cv2.VideoCapture(0)
while True:
    success, imgs = cap.read()
    if not success:
        continue
#    imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    #imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facesCurFrames = face_recognition.face_locations(imgs)
    encodingCurFrames = face_recognition.face_encodings(imgs,facesCurFrames)

    for encodeFace, faceLoc in zip(encodingCurFrames, facesCurFrames):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        face_distance = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(face_distance)
        matchIndex = np.argmin(face_distance)

        if matches[matchIndex]:
            name = class_names[matchIndex]
            print(name)
            #y1,x2,y2,x1 = faceLoc
            left,top,right,bottom = faceLoc
            #cv2.rectangle(imgs, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(imgs, (bottom,left), (top,right), (0, 0, 255), 2)
            #cv2.rectangle(imgs, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            #cv2.rectangle(imgs, (left,top), (right,bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(imgs,name,(bottom,left-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

            #markeAttendance(name)
    cv2.imshow("Face Recognition", imgs)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()