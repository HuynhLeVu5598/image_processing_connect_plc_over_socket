import glob
import os
import cv2
from os import path
import threading
import time
import socket
from time import sleep
import torch
import numpy as np 

soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#soc.settimeout(5)

def socket_connect(host, port):
    try:
        soc.connect((host, port))
        return True
    except OSError:
        print("Can't connect to PLC")
        sleep(3)
        print("Reconnecting....")
        return False
def readdata(data):
    a = 'RD '
    c = '\x0D'
    d = a+ data +c
    datasend = d.encode("UTF-8")
    soc.sendall(datasend)
    data = soc.recv(1024)
    datadeco = data.decode("UTF-8")
    data1 = int(datadeco)

    return data1

#Write data
def writedata(register, data):
    a = 'WR '
    b = ' '
    c = '\x0D'
    d = a+ register + b + str(data) + c
    datasend  = d.encode("UTF-8")
    soc.sendall(datasend)
    datares = soc.recv(1024)
    #print(datares)

def removefile():
    directory1 = 'D:/FH1/'
    directory2 = 'D:/FH2/'
    if os.listdir(directory1) != []:
        for i in glob.glob(directory1+'*'):
            os.remove(i)

    if os.listdir(directory2) != []:
        for i in glob.glob(directory2+'*'):
            os.remove(i)

    directory3 = 'D:/FH/camera1/'
    directory4 = 'D:/FH/camera2/'
    if os.listdir(directory3) != []:
        for i in glob.glob(directory3+'*'):
            for j in glob.glob(i+'/*'):
                os.remove(j)
            os.rmdir(i)

    if os.listdir(directory4) != []:
        for i in glob.glob(directory4+'*'):
            for j in glob.glob(i+'/*'):
                os.remove(j)
            os.rmdir(i)
    print('already delete folder')

def task1():
    if readdata('DM1400')==1000:
        print('1')
        for filename1 in glob.glob('D:/FH/camera1/*'):
            for path1 in glob.glob(filename1 + '/*.jpg'):
                img1 = cv2.imread(path1)
                img1 = cv2.resize(img1,(640,480))   
                cv2.imshow('image1',img1)
                cv2.waitKey(1)    
                os.remove(path1)
            os.rmdir(filename1)
        writedata('DM1500.U',2000)  
        writedata('DM1400.U',2000) 


def task2():
    if readdata('DM2400')==1000:
        print('2')
        for filename2 in glob.glob('D:/FH/camera2/*'):
            for path2 in glob.glob(filename2 + '/*.jpg'):
                img2 = cv2.imread(path2)
                img2 = cv2.resize(img2,(640,480))
                cv2.imshow('image2',img2)
                cv2.waitKey(1)    
                os.remove(path2)
            os.rmdir(filename2)
        writedata('DM2500.U',2000)  
        writedata('DM2400.U',2000) 


def task3():
    a=0
    b=0
    if readdata('DM3400')==1000:
        for filename1 in glob.glob('D:/FH/camera1/*'):
            for path1 in glob.glob(filename1 + '/*.jpg'):
                img1 = cv2.imread(path1)
                img1 = cv2.resize(img1,(640,480))  
                result1 = model(img1,size= size,conf = conf) 
                for i, pred in enumerate(result1.pred):
                    if pred.shape[0]:
                        for *box,cof,clas in reversed(pred):
                            #print(clas)
                            if result1.names[int(clas.tolist())] == 'cup':
                                #print('ok1')
                                writedata('MR102',1)
                                a=1
                                #time.sleep(0.3)
                                #writedata('DM1500.U',2000)  
                                #writedata('DM1400.U',2000) 
                                #writedata('DM3400.U',2000) 
                            else:
                                #print('ng1')
                                writedata('MR103',1)
                                a=1
                                #time.sleep(0.3)
                                #writedata('DM1500.U',2000)  
                                #writedata('DM1400.U',2000) 
                                #writedata('DM3400.U',2000) 
                    else:
                        #print('ng11')
                        writedata('MR103',1)
                        a=1
                        #time.sleep(0.3)
                        #writedata('DM1500.U',2000)  
                        #writedata('DM1400.U',2000) 
                        #writedata('DM3400.U',2000) 
                show1 = np.squeeze(result1.render())
                cv2.imshow('image1',show1)
                cv2.waitKey(1)    
                os.remove(path1)

            os.rmdir(filename1)
            #writedata('DM3400.U',2000) 

        for filename2 in glob.glob('D:/FH/camera2/*'):
            for path2 in glob.glob(filename2 + '/*.jpg'):
                img2 = cv2.imread(path2)
                img2 = cv2.resize(img2,(640,480))
                result2 = model(img2,size= size,conf = conf) 
                for i, pred in enumerate(result2.pred):
                    if pred.shape[0]:
                        for *box,cof,clas in reversed(pred):
                            #print(clas)
                            if result2.names[int(clas.tolist())] == 'cup':
                                #print('ok2')
                                writedata('MR104',1)
                                b=1
                                #time.sleep(0.3)
                                #writedata('DM2500.U',2000)  
                                #writedata('DM2400.U',2000)
                                #writedata('DM3400.U',2000)  
                            else:
                                #print('ng2')
                                writedata('MR105',1)
                                b=1
                                #time.sleep(0.3)
                                #writedata('DM2500.U',2000)  
                                #writedata('DM2400.U',2000) 
                                #writedata('DM3400.U',2000) 
                    else:
                        #print('ng22')
                        writedata('MR105',1)
                        b=1
                        #time.sleep(0.3)
                        #writedata('DM2500.U',2000)  
                        #writedata('DM2400.U',2000) 
                        #writedata('DM3400.U',2000) 
                #print('---')
                show2 = np.squeeze(result2.render())
                cv2.imshow('image2',show2)
                cv2.waitKey(1)    
                os.remove(path2)
            os.rmdir(filename2)
        if a==1 & b==1:
            print('ok')
        else:
            print('error')
        writedata('DM3400.U',2000) 

def task4():
    if readdata('DM3400')==1000:
        for filename2 in glob.glob('D:/FH/camera2/*'):
            for path2 in glob.glob(filename2 + '/*.jpg'):
                img2 = cv2.imread(path2)
                img2 = cv2.resize(img2,(640,480))
                result2 = model(img2,size= size,conf = conf) 
                for i, pred in enumerate(result2.pred):
                    if pred.shape[0]:
                        for *box,cof,clas in reversed(pred):
                            #print(clas)
                            if result2.names[int(clas.tolist())] == 'cup':
                                print('ok2')
                                writedata('MR104',1)
                                time.sleep(0.3)
                                #writedata('DM2500.U',2000)  
                                #writedata('DM2400.U',2000)
                                writedata('DM3400.U',2000)  
                            else:
                                print('ng2')
                                writedata('MR105',1)
                                time.sleep(0.3)
                                #writedata('DM2500.U',2000)  
                                #writedata('DM2400.U',2000) 
                                writedata('DM3400.U',2000) 
                    else:
                        print('ng22')
                        writedata('MR105',1)
                        time.sleep(0.3)
                        #writedata('DM2500.U',2000)  
                        #writedata('DM2400.U',2000) 
                        writedata('DM3400.U',2000) 
                print('---')
                show2 = np.squeeze(result2.render())
                cv2.imshow('image2',show2)
                cv2.waitKey(1)    
                os.remove(path2)
            os.rmdir(filename2)



connected = False
while connected == False:
    connected = socket_connect('192.168.0.20',8501)
print("connected")   

model =torch.hub.load('./levu','custom', path= r'C:\Users\baotri03\Desktop\vu\b03\model.pt', source='local',force_reload =True)
print('model already')

size = 416
conf = 0.4
max_det=1000
classes = 0



removefile()

# while True:
#     t1 = threading.Thread(target=task3)
#     t2 = threading.Thread(target=task4)  

#     t1.start()
#     t2.start()    

#     t1.join()
#     t2.join()    

#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
 
# cv2.destroyAllWindows()


# writedata('MR16000',1) 
# time.sleep(1)
#writedata('DM1600.S',-9) 


while True:
    task3()
    #task4()        
    # writedata('DM1500.U',2000)  
    # writedata('DM1400.U',2000) 
    # writedata('DM2500.U',2000)  
    # writedata('DM2400.U',2000) 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()


