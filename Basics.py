import cv2
import numpy as np
import face_recognition
import PIL.ImageDraw
import PIL.Image

imgShahrukh = face_recognition.load_image_file("ImageBasic/Shahrukh Khan.jpg")
imgShahrukh = cv2.cvtColor(imgShahrukh,cv2.COLOR_BGR2RGB)

imgShahrukhTest = face_recognition.load_image_file("ImageBasic/Shahrukh Khan Test.jpg")
#imgShahrukhTest = face_recognition.load_image_file("ImageBasic/Salman Khan.jpg")
imgShahrukhTest = cv2.cvtColor(imgShahrukhTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgShahrukh)[0]
encode = face_recognition.face_encodings(imgShahrukh)[0]
cv2.rectangle(imgShahrukh,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),color=(255,0,255),thickness=2)
#OR
#left,top,right,bottom = faceLoc
#cv2.rectangle(imgShahrukh,(left,top),(right,bottom),color=(255,0,255),thickness=20)
#OR
#pilImage = PIL.Image.fromarray(imgShahrukh)
#draw = PIL.ImageDraw.Draw(pilImage)
#left,top,right,bottom = faceLoc
#draw.rectangle([left,top,right,bottom],outline='red',width=2)
#pilImage.show()

faceLocTest = face_recognition.face_locations(imgShahrukhTest)[0]
encodeTest = face_recognition.face_encodings(imgShahrukhTest)[0]
cv2.rectangle(imgShahrukhTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),color=(255,0,255),thickness=2)

results = face_recognition.compare_faces([encode],encodeTest)
face_distance = face_recognition.face_distance([encode],encodeTest)
cv2.putText(img=imgShahrukhTest,text="{0}:{1}".format(results,round(face_distance[0],2)),org= (50,50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.5,color=(0,0,255),thickness=1)
print(results,face_distance)
cv2.imshow("Shahrukh",imgShahrukh)
cv2.imshow("Shahrukh Test",imgShahrukhTest)

cv2.waitKey(0)
cv2.destroyAllWindows()