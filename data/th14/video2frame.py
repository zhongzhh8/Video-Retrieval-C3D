import os
import cv2
import numpy as np


def video2frames(video,newdir):

    cap = cv2.VideoCapture(video)
    count=0
    #29.97/30FPS
    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret==True: 
            #cv2.imshow("frame",frame)
            cv2.imwrite(newdir+'/'+str(count).zfill(4)+".jpg",frame)
            #cv2.waitKey(0) 
            count+=1   
        else:
            break
    print count      

#all_pathlabel()       
f=open("test.txt")
l=f.readlines()
f.close()
for item in l:
    video=item.strip().split()[0]
    newdir=item.strip().split()[0].split('.')[0]
    if os.path.exists(newdir)==True:
        os.system("rm -rf "+newdir)
    os.mkdir(newdir)
    video2frames(video,newdir)
