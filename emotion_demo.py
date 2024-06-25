import cv2
from keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
from PIL import ImageFont, ImageDraw, Image 
import numpy as np
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
classifier =load_model('model_optimal.h5')
class_labels = ['Giận dữ', 'Ghê sợ', 'Sợ hãi', 'Hạnh phúc', 'Buồn', 'Bất ngờ', 'Bình thường']
b,g,r,a = 0,255,0,0
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w] # the face after cut from frame
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float')/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi,axis=0)
        preds = classifier.predict(roi)[0]
        label=class_labels[preds.argmax()]
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        draw.text((50, 80), label,  fill = (b, g, r, a))
        print(label)
        frame = np.array(img_pil) #hiển thị ra window    
    cv2.imshow('Emotion Detection',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()