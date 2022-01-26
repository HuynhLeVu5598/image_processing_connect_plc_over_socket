import glob
from msilib.schema import Directory
import os
import cv2
from os import path
import threading
import time
import socket
from time import sleep
import torch
import numpy as np 
import pathlib

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

def take_name_image(directory):
    #directory5 = 'D:/nc/sava_data/'
    if os.listdir(directory) == []:
        i =0
    else:
        path = pathlib.Path(directory)
        # path.stat().st_mtime
        time, file_path = max((f.stat().st_mtime,f) for f in path.iterdir())
        i = int(file_path.stem)
    return i


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


def task3(i):
    if readdata('DM1400')==1000:
        directory3 = 'D:/FH/camera1/'
        if os.listdir(directory3) == []:
            print('folder 1 empty')
        else:
            #a = time.time()
            for filename1 in glob.glob('D:/FH/camera1/*'):
                for path1 in glob.glob(filename1 + '/*.jpg'):
                    img1 = cv2.imread(path1)
                    while type(img1) == type(None):
                        print('error 1')
                        #for filename1 in glob.glob('D:/FH/camera1/*'):
                        for path1 in glob.glob(filename1 + '/*.jpg'):
                            img1 = cv2.imread(path1)
                    cv2.imwrite('D:/nc/save_data5/' + str(i) + '.jpg',img1)
                    #i+=1
                    img1 = cv2.resize(img1,(640,480))
                    result1 = model(img1,size= size,conf = conf) 
                    for i0, pred in enumerate(result1.pred):
                        if pred.shape[0]:
                            for *box,cof,clas in reversed(pred):
                                #print(clas)
                                if result1.names[int(clas.tolist())] == 'bactruc':
                                    print('ok1')
                                    #writedata('MR102',1)
                                    #time.sleep(0.3)
                                    writedata('DM1500.U',2000)  
                                    writedata('DM1400.U',2000) 
                                else:
                                    print('ng1')
                                    #writedata('MR103',1)
                                    #time.sleep(0.3)
                                    writedata('DM1500.U',2000)  
                                    writedata('DM1400.U',2000)
                        else:
                            print('ng11')
                            #writedata('MR103',1)
                            #time.sleep(0.3)
                            writedata('DM1500.U',2000)  
                            writedata('DM1400.U',2000) 
                    show1 = np.squeeze(result1.render())
                    cv2.imshow('image1',show1)
                    #b = time.time() - a
                    #print(str(b))
                    cv2.waitKey(1)    
                    os.remove(path1)

                os.rmdir(filename1)


def task4(j):
    if readdata('DM2400')==1000:
        directory4 = 'D:/FH/camera2/'
        if os.listdir(directory4) == []:
            print('folder 2 empty')
        else:
            for filename2 in glob.glob('D:/FH/camera2/*'):
                for path2 in glob.glob(filename2 + '/*.jpg'):
                    img2 = cv2.imread(path2)
                    while type(img2) == type(None):
                        print('error 2')
                        #for filename2 in glob.glob('D:/FH/camera2/*'):
                        for path2 in glob.glob(filename2 + '/*.jpg'):
                            img2 = cv2.imread(path2)
                    #img2 = cv2.resize(img2,(640,480))
                    #j+=1
                    cv2.imwrite('D:/nc/save_data6/' + str(j) + '.jpg',img2)
                    #j+=1
                    img2 = cv2.resize(img2,(640,480))
                    result2 = model(img2,size= size,conf = conf) 
                    for i0, pred in enumerate(result2.pred):
                        if pred.shape[0]:
                            for *box,cof,clas in reversed(pred):
                                #print(clas)
                                if result2.names[int(clas.tolist())] == 'bactruc':
                                    print('ok2')
                                    #writedata('MR104',1)
                                    #time.sleep(0.3)
                                    writedata('DM2500.U',2000) 
                                    writedata('DM2400.U',2000) 
                                    print('---')
                                else:
                                    print('ng2')
                                    #writedata('MR105',1)
                                    #time.sleep(0.3)
                                    writedata('DM2500.U',2000)  
                                    writedata('DM2400.U',2000) 
                                    print('---')
                        else:
                            print('ng22')
                            #writedata('MR105',1)
                            #time.sleep(0.3)
                            writedata('DM2500.U',2000)  
                            writedata('DM2400.U',2000) 
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
#model =torch.hub.load('./levu','custom', path= 'C:/Users/baotri03/Desktop/vu/1/yolov5s.pt', source='local',force_reload =True, device='cpu')
print('model already')

size = 416
conf = 0.4
max_det=1000
classes = 0

removefile()
i = take_name_image('D:/nc/save_data5/')
j = take_name_image('D:/nc/save_data6/')

print(str(i))
print(str(j))
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
    # a=0
    # b=0
    #print('***')
    task3(i)
    task4(j) 
    i+=1
    j+=1

    #print('---')  
    # if a==1 & b==1:
    #     print('two folder')
    # else:
    #     print('one folder')

    # writedata('DM1500.U',2000)  
    # writedata('DM1400.U',2000) 
    # writedata('DM2500.U',2000)  
    # writedata('DM2400.U',2000) 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()


