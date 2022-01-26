import glob
import os
import cv2
from os import path
from time import sleep
import torch
import numpy as np 
import time

def removefile():
    directory1 = 'D:/FH1/'
    directory2 = 'D:/FH2/'
    if os.listdir(directory1) != []:
        for i in glob.glob(directory1+'*'):
            os.remove(i)

    if os.listdir(directory2) != []:
        for i in glob.glob(directory2+'*'):
            os.remove(i)


def task5():
    #print('1')
    for filename1 in glob.glob('D:/FH/camera1/*'):
        for path1 in glob.glob(filename1 + '/*.jpg'):
            img1 = cv2.imread(path1)
            img1 = cv2.resize(img1,(640,480))  
            result1 = model(img1,size= size,conf = conf) 
            for i, pred in enumerate(result1.pred):
                if pred.shape[0]:
                    for *box,cof,clas in reversed(pred):
                        #print(clas)
                        if result1.names[int(clas.tolist())] == 'person':
                            print('ok1')
                        else:
                            print('ng1')
                else:
                    print('ng11')
            show1 = np.squeeze(result1.render())
            cv2.imshow('image1',show1)
            cv2.waitKey(1)    
            os.remove(path1)
        os.rmdir(filename1)

def task6():
    #print('2')
    for filename2 in glob.glob('D:/FH/camera2/*'):
        for path2 in glob.glob(filename2 + '/*.jpg'):
            img2 = cv2.imread(path2)
            img2 = cv2.resize(img2,(640,480))
            previous = time.time()
            result2 = model(img2,size= size,conf = conf) 
            for i, pred in enumerate(result2.pred):
                if pred.shape[0]:
                    for *box,cof,clas in reversed(pred):
                        #print(clas)
                        if result2.names[int(clas.tolist())] == 'person':
                            print('ok2')
                        else:
                            print('ng2')
                else:
                    print('ng22')
            show2 = np.squeeze(result2.render())
            cv2.imshow('image2',show2)
            t = time.time()-previous
            print(str(t))
            cv2.waitKey(1)    
            os.remove(path2)
        os.rmdir(filename2)




model =torch.hub.load('./levu','custom', path= 'C:/Users/baotri03/Desktop/vu/1/yolov5s.pt', source='local',force_reload =True)
print('model already')
size = 416
conf = 0.4
max_det=1000
classes = 0


removefile()

# while True:
# # #     # t1 = threading.Thread(target=task1)
# # #     # t2 = threading.Thread(target=task2)  

# # #     # t1.start()
# # #     # t2.start()    

# # #     # t1.join()
# # #     # t2.join()    
# #     num=0
#     task1()
#     task2()
#     if cv2.waitKey(60000):
#         break
 
# cv2.destroyAllWindows()


# writedata('MR16000',1) 
# time.sleep(1)
# writedata('MR16000',0) 


while True:
    task5()
    task6()
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()


