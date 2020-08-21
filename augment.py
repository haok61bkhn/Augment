import imgaug.augmenters as iaa
import cv2
import numpy as np
import uuid
import glob
import random
class Augment(object):
    def __init__(self,ratio):
        self.ratio = ratio

    def aug1(self,img):
        seq = iaa.Sequential([
            iaa.MaxPooling(kernel_size=2) #=2 or 3
        ])
        img_au=seq(image=img)
        return img_au

    def aug2(self,img):
        x= random.randint(2,3)
        
        seq = iaa.Sequential([
            iaa.MinPooling(kernel_size=x) #=2 or 3,4,5
        ])
        img_au=seq(image=img)
        return img_au

    def aug3(self,img):
        seq=iaa.Sequential([iaa.Fog()])
        img_au=seq(image=img)
        return img_au
    
    def aug4(self,img):
        seq=iaa.Sequential([
        iaa.Rain() 
        ])
        img_au=seq(image=img)
        return img_au
    
    def run(self,img):
        img_aus=[]
        x=random.randint(0,100)
        if(x/100>self.ratio):
            img_aus.append(self.aug1(img))

        x=random.randint(0,100)
        if(x/100>self.ratio):
            img_aus.append(self.aug2(img))

        x=random.randint(0,100)
        if(x/100>self.ratio):
            img_aus.append(self.aug3(img))

        x=random.randint(0,100)
        if(x/100>self.ratio):
            img_aus.append(self.aug4(img))
        
        return img_aus


if __name__ == "__main__":
    X=Augment(0.5)
    img=cv2.imread("data/0_423495668.jpg")
    cv2.imshow("orignal",img)
    img_aus=X.run(img)
    for id,im in enumerate(img_aus):
        cv2.imshow(str(id),im)
    cv2.waitKey(0)
