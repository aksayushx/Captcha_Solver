import cv2
import os
import segmentation as seg
from tensorflow.keras.models import load_model
import numpy as np

model=load_model('model.h5',compile=True)
classes='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    
def predict(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img_paths=seg.extract_character(image)
    output=' '
    for i in img_paths:
        m=[]
        img=cv2.imread(i,cv2.IMREAD_GRAYSCALE)
        img = cv2.bitwise_not(img)
        img=np.reshape(img,(28,28,1))/255
        m.append(img)
        m=np.array(m)
        result=np.argmax(model.predict(m))
        output+=classes[result]
    return output
        
    
def test():
    #Enter filenames to be tested in image_paths after adding them to this folder
    image_paths=['5.jpeg','6.jpeg']
    for i in image_paths:
        image=cv2.imread(i)
        captcha_decoded=predict(image)
        print(captcha_decoded)

if __name__=='__main__':
    test()

