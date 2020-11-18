import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

model = model_from_json(open("fer.json","r").read())
model.load_weights("fer_weights.h5")

face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    success,imsgs = cap.read()
    if not success:
        continue

    gray_img = cv2.cvtColor(imsgs,cv2.COLOR_BGR2GRAY)
    face_detected = face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)

    for x,y,w,h in face_detected:
        cv2.rectangle(imsgs,(x,y),(x+w,y+h),(255,0,0),2)
        img_gray = gray_img[y:y+w,x:x+h]
        img_gray = cv2.resize(img_gray,(48,48))
        img_pixels = image.img_to_array(img_gray)
        img_pixels = np.expand_dims(img_pixels,axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        max_index = np.argmax(predictions[0])
        emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotion[max_index]
        cv2.putText(imsgs,predicted_emotion,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0,),3)


    cv2.imshow("Facial Emotion Analysis", imsgs)
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()