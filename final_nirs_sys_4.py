# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 08:53:05 2015

@author: bci2000
"""


import os
current_dir=os.getcwd()
os.chdir('C:\\libsvm-3.20\\python')#set path
from svmutil import *
from ctypes import *
from ctypes import windll
import math
import stop
import numpy as np
import time
import ImageTk
import Image
from scipy  import signal
from operator import itemgetter
from Tkinter import *
import wave
import winsound 
import random
import threading 
import sys
os.chdir('C:\NIRx\NIRx SDK package 2011-08-15\TomographyMATLAB_API\LibraryM\TomoAPI')

objdll=cdll.LoadLibrary('TomographyAPI')
#import serial
os.chdir(current_dir)

def sendtrigger(trigger): #this function maybe need to change to adapt to your system
    p=windll.inpout32
    p.Out32(0x2CF8,trigger)
    time.sleep(0.004)
    p.Out32(0x2CF8,0)

def initial(address, port, timeout):
#    os.chdir('C:\NIRx\NIRx SDK package 2011-08-15\TomographyMATLAB_API\LibraryM\TomoAPI')
#
#    objdll=cdll.LoadLibrary('TomographyAPI')   
    error_out = objdll.tsdk_initialize()
    #stop()
    print 'Initialized!\n'
    if error_out!=0:
        print 'Connection Closed\n'
        
        
        objdll.tsdk_close()
        
    #Initialize the API    
    error_out = objdll.tsdk_connect(address, port, timeout)    
    
    
    #                 Connect to server
    ###Check the server status
    
    statusFlag0 =(c_uint*1)()
    sampleRate=(c_double*1)()
    
    print statusFlag0[0]
    print sampleRate[0]
    error_out1= objdll.tsdk_getStatus(statusFlag0,sampleRate)
    print statusFlag0[0]
    print sampleRate[0]
    
    ###Check desired parameters names
    
    pName=(c_char*1)()
    print pName
    error_out2= objdll.tsdk_getName(0, 0, pName[0], 20)
    print pName
    
    
    ####Retrieve the number of available channels
    numSor = (c_uint*1)()
    numDet =(c_uint*1)()
    numWav =(c_uint*1)()
    print numSor[0], numDet[0], numWav[0]
    error_out3= objdll.tsdk_getChannels(numSor, numDet, numWav)
    print numSor[0], numDet[0], numWav[0]
    frameSize=(c_uint*1)()
    frameSize[0] = numSor[0]*numDet[0]*numWav[0]
    
    ##Ask the server to start streaming
    
    sources=(c_int32*1)()
    detectors=(c_int32*1)()
    wavelengths=(c_int32*1)()
    error_out4=objdll.tsdk_start(sources, detectors, wavelengths, numSor[0], numDet[0], numWav[0], frameSize)
    elementsPerFrame=frameSize
    print sources, detectors, wavelengths, elementsPerFrame[0]
    print 'API prepared to receive data.'
    return numSor, numDet, numWav
#def stop():
#    API.tsdk_stop()
#    API.tsdk_disconnect()
#    API.tsdk_close()

def getdata2(numSor, numDet, numWav,framecount,timeout,newdata,Time2):
    reqFrames = c_int(1)

    timestamps=(c_double*1)()
    timingBytes=(c_char*1)()
    frameSize=(c_int32*1)()
    frameSize[0] = numSor[0]*numDet[0]*numWav[0]
    buffersize=(c_int*1)(reqFrames.value * frameSize[0])
    data =(c_float*buffersize[0])()
    frameCount=(c_int*1)()
    while framecount.value:
        
        error_out= objdll.tsdk_getNFrames(reqFrames.value,timeout.value,frameCount, timestamps, timingBytes, data, buffersize)
#        print frameCount[0], timestamps[0], ord(timingBytes[0]), data[:], dataBufferSize[0]
        framecount.value=framecount.value-1
        Data=np.array(data)
        pr=open(Time2+'dataset.txt','a')
        pr.write(str(Data)+'\n') 
#        pr.write('\n')
        pr.close()	
        newdata.append(data[:])
#        print len(newdata)

   
    newdata=[newdata]
    return newdata,timestamps,frameCount

class item:
    def __init__(self):
        self.wl = [760,850]
        self.LBsf=6.2500
        self.LBodistance= 2.5000
        self.LBlpfreq= 0.5
        self.LBdpf= [5.9800,7.1500]
        self.LBepsilon = [[1.4866,3.8437], [2.5264,1.7986]]
        self.LBmode=1  
ni=item()
def onlineLBG2(ni,dat):
    cc_oxy=[]
    cc_deo=[]
    e=ni.LBepsilon
   
    DPF=ni.LBdpf
    Loptoddistance=ni.LBodistance
    lpfreq=ni.LBlpfreq
    sf=ni.LBsf
    
    
    s1=dat.shape[0]
    s2=dat.shape[1]
    
    dat=dat+2.2204e-16
    a=np.array([[row[i] for i in range(0,s2/2)] for row in dat])#the first half column of dat
    bl_loweWL=[np.sum(row[:])/len(row) for row in a.T]# the mean value of each column
    
    b=np.array([[row[i] for i in range(s2/2,s2)] for row in dat])#the first half column of dat
    bl_highWL=[np.sum(row[:])/len(row) for row in b.T]# the mean value of each column
   
    """ filter the data"""    
    lk = lpfreq*2/sf
    m,n=signal.butter(3,lk,'low')
    lp_loweWL=signal.filtfilt(m,n,a)#higher Wavelength array
    lp_highWL=signal.filtfilt(m,n,b)#lower Wavelength  array
    
    
    normHighWL= lp_highWL/ bl_highWL
    normlowWL= lp_loweWL/ bl_loweWL
    
    Att_highWL=[]
    Att_loweWL=[]
    for row in normHighWL:
        for i in range(0,s2/2):
            if row[i]<0:
                row[i]=2.2204e-16
            Att_highWL.append((-1*math.log10(row[i])).real)
    Att_highWL=np.reshape(np.array(Att_highWL),[s1,s2/2])
     
    for row in normlowWL:
        for i in range(0,s2/2):
            if row[i]<0:
                row[i]=2.2204e-16
            Att_loweWL.append((-1*math.log10(row[i])).real)
    Att_loweWL=np.reshape(np.array(Att_loweWL),[s1,s2/2])     
    
    
#    Att_highWL=np.array([[(-math.log10(row[i])).real for i in range(0,s2/2)] for row in normHighWL])   
#    Att_loweWL=np.array([[(-math.log10(row[i])).real for i in range(0,s2/2)] for row in normlowWL])
        
    if True:
      e=np.array([[row[i]/10 for i in range(0,2)] for row in e])
      c=np.reshape(np.array(DPF+DPF),[2,2])
      e2=   e* c.T *Loptoddistance
#      print e2
      for j in range(0,s2/2):
          a1=[[itemgetter(j)(i) for i in Att_highWL]]
          a1.append([itemgetter(j)(i) for i in Att_loweWL])
          a1=np.array(a1)
          dd=np.dot(np.linalg.inv(e2),a1)
          cc_oxy.append(dd[0])
          cc_deo.append(dd[1])
    bb=cc_oxy+cc_deo
    aa=np.array(bb)
    return aa,cc_oxy,cc_deo
def nirsfeedback(bdata,tdata,testfeature,blocktype):
    """online feedback function"""
#    global numcount1
#    numcount1=numcount1+1

    l1=tdata.shape[0]
    l2=tdata.shape[1]
#    bdata=float(bdata)
#    tdata =float(tdata)               
    mbaseline=[np.sum(row[:])/len(row) for row in bdata]# the mean value of each row (channel)
    tdata1= [tdata[i]-np.ones(tdata[i].shape)*mbaseline[i] for i in range(0,l1)]
    if blocktype == 1:
        testfeature=[]        
    testfeature.append([np.sum(row[:])/len(row) for row in tdata1])
#    testfeature+=([np.sum(row[:])/len(row) for row in tdata1])
    return testfeature
    
def scale(Data): 
    Lower = 0
    Upper = 1
    
    data=np.array(Data)
    
    R=data.shape[0]# num samples 5
    C=data.shape[1]#num features 6

    k=data.T
    MaxV=[max(i) for i in k]
    MinV=[min(i) for i in k]
    
    


    scaled=(data-np.dot(np.ones([R,1]),MinV*np.ones([1,C])))*(np.ones([R,1])*((Upper-Lower)*np.ones([1,C])/(np.array(MaxV)-np.array(MinV))))+Lower
    return scaled ,MaxV,MinV
def scaleproj(Data, model_MaxV,model_MinV):
    Lower = 0
    Upper = 1
    data=np.array(Data)
    
    R=data.shape[0]# num samples 
    C=data.shape[1]#num features 

#    k=data.T
#    MaxV=[max(i) for i in k]
#    MinV=[min(i) for i in k]
    MaxV=model_MaxV
    MinV=model_MinV

    scaled=(data-np.dot(np.ones([R,1]),MinV*np.ones([1,C])))*(np.ones([R,1])*((Upper-Lower)*np.ones([1,C])/(np.array(MaxV)-np.array(MinV))))+Lower
    return scaled 
class BCI(threading.Thread):
    def __init__(self, t_name,soundNum):
        threading.Thread.__init__(self, name=t_name)
        self.soundNum=soundNum
    def run(self):
        finishflag=False# the current run finish flag
        global start_flag #  flag to determine when to play sound
        global testfeature#  data features
        global svm_model
        global current_dir#current dir
        start_flag=0
        senten_flag=True
        global Time2
        global root
#        Time=time.strftime('%Y-%m-%d',time.localtime(time.time()))  #current time
#        Time2=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))# for current data file name
#        global p_acc_cal
#        numcount1=0# the number of trail in current run,start from 0
        """timeout  no need """
        timeout=c_int(5000)

        bframecount=c_int(32)

        dframecount=c_int(98)
        #blocktype=0
        """the statistic of the result"""
        totalcorrect = 0
        totalincorrect = 0



        classlabel=[]
        tempdata=[]
        newdata=[]
        newdata1=[]# to save baseline data
        newdata2=[]# to save reponde data

        baselinedata=[]
        baselinelabel=[]

        NIRSdata=[]
        datlabel=[]

        testfeature=[]
        x=[]#the data used to do the svm classification
#        num=0
        tri=[]
        p_labels1=[]#to save the feedback_predicted labels
        """connect the NIRS device"""
        numSor, numDet, numWav=initial('192.168.0.102', 45342, 5000)
#        print numSor[0], numDet[0], numWav[0] 
        reqFrames = c_int(1)         
        """the framesize of one sample"""
        frameSize=(c_int32*1)()
        frameSize[0] = numSor[0]*numDet[0]*numWav[0]
        buffersize=(c_int*1)(reqFrames.value * frameSize[0])
#        print buffersize[0]
        
  
        
        while not finishflag:#for the finish of current run
#            if start_flag==0:
            global count#the number of current RUN
            global numcount1#the number of current trails
            if count<calibrationnum:
                blocktype=0          
            else:
                blocktype=1                

      
		
     
            """the default paramater for NIRSport"""
            acq=True
		
            
            while acq:
  
  
                """monitor cycle"""
                frameCount =(c_int*1)()
        
                timestamps=(c_double*1)()

                timingBytes=(c_char*1)()
                data =(c_float*buffersize[0])()

                error_out= objdll.tsdk_getNFrames(reqFrames.value,timeout.value,frameCount, timestamps, timingBytes, data, buffersize)

                dataBufferSize=buffersize
                
                
                if senten_flag==True:
                    start_flag=1
                    senten_flag=False
                
    #    print frameCount[0], timestamps[0], timingBytes, tempdata[:], dataBufferSize[0]
#                print timingBytes.value
                """get the trigger"""
#                while senten_flag:
                if timingBytes.value!='':
                    
                    tB_i=ord(timingBytes[0])
            
            
                    print 'the label is' + ' '+str(tB_i)
            
                    if(tB_i==baslitrig[0] or tB_i==baslitrig[1]):
                        start=time.time()
                        """
                        getdata2
                        """
#                        print numSor[0], numDet[0], numWav[0],bframecount,timeout,newdata
                        getdata2(numSor, numDet, numWav,bframecount,timeout,newdata,Time2)

                        newdata1.append(newdata)
                
                
#                        print newdata
                        bdata=np.array(newdata)#array ,single data
#                        print type(bdata)
#                        print ';;;;;;;;;'
#                        print bdata.shape[0]

                        bdata,cc_oxy1,cc_deo1=onlineLBG2(ni,bdata)
                        print 'done_bdata'
                
                
                
    #            tempdata.append(bdata)
                
                        baselinedata=bdata#2,32,40
                        baselinelabel.append(tB_i)
                
                        newdata=[]
                        bframecount=c_int(32)
                
    #            print tempdata[:],ttimestamps[:]
    #            tempdata=[tempdata[:]]
                        finished=time.time()
                        print 'Elapsed time is' +  ' '+str(finished-start)+ 'seconds'
#                        start_flag=1
                
                
                
                    elif(tB_i==datrig[0] or tB_i==datrig[1]):
                        start=time.time()
                        """
                        getdata2
                        """
#                        numcount1+=1
                        getdata2(numSor, numDet, numWav,dframecount,timeout,newdata,Time2)

                        newdata2.append(newdata)                                                
                        tdata=np.array(newdata)#array ,single data
                        
                        tdata,cc_oxy2,cc_deo2=onlineLBG2(ni,tdata)
                        
                        print 'done_tdata'
    #            
    #            tempdata.append(tdata)
    #            
    #            NIRSdata=tempdata
                        datlabel.append(tB_i)
                
                        for element in range(classnum):
                            if tB_i==datrig[element]:
                                classlabel.append(element-1)
                                tri+=[float(element-1)]
                        newdata=[]
                        dframecount=c_int(98)
    #            print tempdata[:],ttimestamps[:]
    #            tempdata=[tempdata[:]]
    
                
                        testfeature=nirsfeedback(bdata,tdata,testfeature,blocktype)# features
                        """belows are the data processing and model trained process"""
                        if blocktype == 0 and numcount1==2*self.soundNum:
                            if count!=0:
                                arr=[]
                                brr=[]
                                for i in range(1,count+1):
                                    feature_file=Time2+'clisiffication'+str(i)+'.txt'
                                    fp=open(feature_file)
                                    for lines in fp.readlines():
                                        lines=lines.replace("\n","").split(" ")
                                        lines=[float(j) for j in lines]
                                        arr.append(lines)
                                    fp.close()
                                    label_file=Time2+'datalabel'+str(i)+'.txt'
                                    fp=open(label_file)
                                    for lines in fp.readlines():
                                        lines=lines.replace("\n","").split(" ")
                                        lines=[float(k) for k in lines]
                                        brr.extend(lines)
                                    fp.close()
                                    
                                    testfeature=testfeature+arr
                                    tri=tri+brr                   
                            testfeature,model_MaxV,model_MinV=scale(testfeature)
                            model_pra=[model_MaxV,model_MinV]

                    
                    
                            a=np.array(testfeature)
                            l1=a.shape[0]
                            l2=a.shape[1]
                            print l1,l2

                
                            """
                            This is used to change the data to the libsvm form
                            """
                            for i in range(0,l1):
                                dict={}
                                for j in range(0,l2):                            
                                    dict[j+1]=float(a[i][j])
                                x=x+[dict]

                            """get svm_model"""
                            print tri,type(x)
                            svm_model = svm_train(tri,x,['-t',0])#train svm_model
#                            svm_save_model('svm_model', svm_model)  
                            """load svm_model and predict"""  
#                            svm_model=svm_load_model('svm_model')
#                            global p_acc_cal
                            p_labels,p_acc_cal,p_vals=svm_predict(tri,x,svm_model)
                            print p_acc_cal
                            
                    
                        elif blocktype == 1:
                            model_file=Time2+'model'+'.txt'
                            fp=open(model_file)
                            for lines in fp.readlines():
                                lines=lines.replace("\n","").split(" ")
                                lines=[float(j) for j in lines]
                            model_MaxV=lines[0]
                            model_MinV=lines[1]
                            testfeature=scaleproj(testfeature,model_MaxV,model_MinV)
#                            testfeature,MaxV,MinV=scale(testfeature)
              
    #                """see if this is needed"""
                            a=np.array(testfeature)
                            l1=a.shape[0]
                            l2=a.shape[1]
        
                            """
                            This is used to change the data to the libsvm form
                            """
                            for i in range(0,l1):
                                dict={}
                                for j in range(0,l2):
                            
                                    dict[j+1]=float(a[i][j])
                                x=x+[dict]
                            
                       
                            """load svm_model and predict"""  
#                            svm_model=svm_load_model('svm_model')
                            print len(tri),len(x)

                            p_labels,p_acc_feb,p_vals=svm_predict(tri,x,svm_model)
#                            pr=open(Time2+'predictedlabels'+str(count+1)+'.txt','a')
#                            pr.write(str(p_labels)+'\n') 
#                    #        pr.write('\n')
#                            pr.close()
#                            p_labels1.append(p_labels)
                            
                            if p_labels==tri:
                                totalcorrect+=1
                                print totalcorrect
                                soundfile=current_dir+'\\Deine Antwort wurde als JA erkannt.wav'
                                winsound.PlaySound(soundfile, winsound.SND_FILENAME|winsound.SND_ASYNC)
                                f = wave.open(soundfile, "rb")
#                                params = f.getparams()
                                frames = f.getnframes()
                                framerate = f.getframerate()		    
                                timetrue = float(frames)*(1.0 / framerate)	  #The time of true sentence 		    
                                time.sleep(timetrue)
                                print 'This is right'
                            else:
                                totalincorrect+=1
                                print totalincorrect
                                soundfile=current_dir+'\\Deine Antwort wurde als NEIN erkannt.wav'
                                winsound.PlaySound(soundfile, winsound.SND_FILENAME|winsound.SND_ASYNC)
                                f = wave.open(soundfile, "rb")
#                                params = f.getparams()
                                frames = f.getnframes()
                                framerate = f.getframerate()		    
                                timetrue = float(frames)*(1.0 / framerate)	  #The time of true sentence 		    
                                time.sleep(timetrue)
                                print 'This is wrong'
                            
                           
                                
                                
                        finished=time.time()
                        print 'Elapsed time is' +  ' '+str(finished-start)+ 'seconds'        
                        start_flag=1 
                        
            
                    elif tB_i==finish:
                        acq=False
                        stop.stop()
                        
                    
                
            
    
#            global p_acc_cal    
            if blocktype == 1 and tri != 2 and numcount1==2*self.soundNum:
                totalresult = totalcorrect/(totalincorrect+totalcorrect)*100
                print 'The total accuracy is ' +' '+ str(p_acc_feb[0])+ '%'
            if blocktype == 0 and tri != 2 and numcount1==2*self.soundNum:
                
                print 'The total accuracy is ' + ' '+str(p_acc_cal[0])+ '%'
      

                #receive data 
                #process data
		
            np.savetxt(Time2+'datalabel'+str(count+1)+'.txt',tri)
            np.savetxt(Time2+'clisiffication'+str(count+1)+'.txt',testfeature)
            
            if count==0:
                np.savetxt(Time2+'model'+'.txt',model_pra)
                np.savetxt(Time2+'predictedlabels'+str(count+1)+'.txt',p_labels)
            else:
                np.savetxt(Time2+'predictedlabels'+str(count+1)+'.txt',p_labels)
                
                
            if numcount1==2*self.soundNum and acq==False:

                
                if count<calibrationnum:
                    cv.itemconfig(rt,text='The accuracy is '+str(p_acc_cal[0]),fill='blue')
                    cv.update()	
                    time.sleep(2)
                    cv.itemconfig(rt,text='',fill='blue')
                    button['state']='active'
                    button['text']='Calibration'
                    if count==calibrationnum-1:
                        button['text']='Feedback'
#                    blocktype=runtypes[count-1]	
#                    print blocktype
		    
                elif count>=calibrationnum and count<totalrunnum:
                    cv.itemconfig(rt,text='The accuracy is '+str(p_acc_feb[0]),fill='blue')
                    cv.update()	
                    time.sleep(2)  #execute it to display the text 'The accuracy is'
                    cv.itemconfig(rt,text='',fill='blue')	
                    button['state']='active'
                    button['text']='Feedback'
#                    blocktype=runtypes[count-1]
#                    print blocktype
		    
                if count==totalrunnum-1:
                    
                    button['state']='disable'
                    button['text']='Finish'	
#                    button['command']=quit
                    
                    
                    
                    
                finishflag=True
                count+=1	
                numcount1=0	
#            np.savetxt(Time+'basiline_data'+str(count+1)+'.txt',newdata1)
#            np.savetxt(Time+'reponse_data'+str(count+1)+'.txt',newdata2)

                        


    

class playsound(threading.Thread):
    def __init__(self, t_name,count,soundNum,numcount1): 
        threading.Thread.__init__(self, name=t_name)
        self.count=count
        self.numcount1=numcount1
        self.soundNum=soundNum
    def run(self):
        flag=False
        numTrue=0
        numFalse=0
        global Time2
        global current_dir
#        global numcount1
	
	   #one session have 10 true sentences and 10 false sentences	
        nameTrue10=nametrue40[self.count*soundNum:(self.count+1)*soundNum]  #In one session playing these 10 true sentences
        nameFalse10=namefalse40[self.count*soundNum:(self.count+1)*soundNum]  #In one session playing these 10 true sentences			
		
	   #generate a list that without 3 number in a row are same
        labelTrue=1;labelFalse=2
	
        soundList=[]
        soundTrue=self.soundNum*[labelTrue]  #soundTrue=[1,1,1,1,1,1,1,1,1,1]
        soundFalse=self.soundNum*[labelFalse]  #soundFalse=[2,2,2,2,2,2,2,2,2,2]
        soundList=soundTrue+soundFalse  #soundList=[1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2]
        limitTrue=3*[labelTrue]
        limitFalse=3*[labelFalse]
        random.shuffle(soundList)
        limitTrue=''.join(map(str,limitTrue))
        limitFalse=''.join(map(str,limitFalse))
        while (limitFalse in ''.join(map(str,soundList)) )or (limitTrue in ''.join(map(str,soundList))):
	       random.shuffle(soundList)
        print soundList	
#        Time=time.strftime('%Y-%m-%d',time.localtime(time.time()))  #current time     
        while not flag:
            global start_flag
#            sendtrigger(0x7)
            if start_flag==1: 
                
                global numcount1
                if numcount1<2*self.soundNum:
                    if count<calibrationnum:
    #                    blocktype=0
                        cv.itemconfig(rt,text='Run'+str(count+1) +':Calibration\n\n\n'+'   Trial:        '+str(numcount1+1),fill='blue')
                        cv.update()
                  
                    elif count>=calibrationnum and count<totalrunnum:
    #                    blocktype=1                
                        cv.itemconfig(rt,text='Run'+str(count+1) +':Feedback\n\n\n'+'   Trial:        '+str(numcount1+1),fill='blue')
                        cv.update()
    		
                    #sendtrigger(0x4)  #display the trigger S 4
                    if soundList[numcount1]==1:				    				    
                        soundfile=current_dir+"\\True\\"+nameTrue10[numTrue]+'.wav'
                        print 'This sentence is'+' '+str(nameTrue10[numTrue]) 
                        sendtrigger(baslitrig[0])
                        time.sleep(4)
    #                    sendtrigger(baslitrig[2])
                        winsound.PlaySound(soundfile, winsound.SND_FILENAME|winsound.SND_ASYNC)
                        f = wave.open(soundfile, "rb")
                        params = f.getparams()
                        frames = f.getnframes()
                        framerate = f.getframerate()		    
                        timetrue = float(frames)*(1.0 / framerate)	  #The time of true sentence 		    
                        time.sleep(timetrue)
                        sendtrigger(datrig[0])
                        start_flag=0  
#                        if numcount1==0 and count==0:
#                            os.path.exists(Time+'sentencelist.txt')
                        pr=open(Time2+'sentencelist.txt','a') 
                        if numcount1==0:
                            pr.write('Run'+str(count+1)+':'+nameTrue10[numTrue]+',')	#add the name of sentence to the txt
                        else:
                            pr.write(nameTrue10[numTrue]+',')
                        pr.close()
                        numTrue=numTrue+1 
    
                    elif soundList[numcount1]==2 :    #if sentense is false		    		    
                        soundfile=current_dir+"\\False\\"+nameFalse10[numFalse]+'.wav'
                        print 'This sentence is'+' '+str(nameFalse10[numFalse])
                        sendtrigger(baslitrig[1])
                        time.sleep(4)
    #                    sendtrigger(datrig[2])
                        winsound.PlaySound(soundfile, winsound.SND_FILENAME|winsound.SND_ASYNC)  
                        f = wave.open(soundfile, "rb")
                        params = f.getparams()
                        frames = f.getnframes()
                        framerate = f.getframerate()		    
                        timefalse = float(frames) * (1.0 / framerate)   # compute the time of one sentence that is playing           		    
                        time.sleep(timefalse)
                        sendtrigger(datrig[1])
                        start_flag=0                        
                        pr=open(Time2+'sentencelist.txt','a')
                        if numcount1==0:
                            pr.write('Run'+str(count+1)+':'+nameFalse10[numFalse]+',')
                        else:                    
                            pr.write(nameFalse10[numFalse]+',')	   # add the name of sentence to the txt
                        pr.close()
                        numFalse=numFalse+1
    
                    numcount1+=1
                    print numcount1
                elif numcount1==2*self.soundNum:
                    print numcount1
                    
                    
                    flag=True
                    pr=open(Time2+'sentencelist.txt','a')
                    pr.write('\n')
                    pr.close()	
#                    time.sleep(5)
                    sendtrigger(finish)
                    start_flag=0


def playing(count,numcount1):
    bci=BCI('BCI',soundNum)
    sound=playsound('playsound',count,soundNum,numcount1)
    bci.start()
    sound.start()
    

def buttoncallback():
    global count
    global numcount1
#    numcount1=0
    if count<calibrationnum:
	
        button['state']='disabled'
        cv.itemconfig(rt,text='Run'+str(count+1)+':calibration',fill='blue')   
        cv.update()
        playing(count,numcount1) #change to your calibration experiment
    elif count>=calibrationnum and count<totalrunnum:
	button['state']='disabled'
	cv.itemconfig(rt,text='Run'+str(count+1)+':feedback',fill='blue') 
	cv.update()
	playing(count,numcount1)   #change to your feedback experiment 1	

es=open('experimentsequence.txt')
runtypes=[]
for line in es.readlines():
    linearr=line.strip().split()
    linearr=[int(i) for i in linearr]
    runtypes.extend(linearr)
calibrationnum=runtypes.count(0)

feedbacknum=runtypes.count(1)
totalrunnum=calibrationnum+feedbacknum
soundNum=1   #2*soundNum is the number of sentences per run
count=0
numcount1=0
Time2=[]
baslitrig=[0xa,0xb]
datrig=[0x4,0x8]
finish=0xf
nametrue=[];namefalse=[]
aa=[]
bdata=[]
svm_model=[]
#baslitrig=[1,2]
#datrig=[4,8]
classnum=len(datrig)
#finish=3




#choose 40 false sentences and 40 true sentences randomly
audio_files_nametrue=[os.path.split(f)[1].split(".") for f in os.listdir("True") \
                      if os.path.split(f)[1].split(".")[1] == 'wav']
for i in range(len(audio_files_nametrue)):
    nametrue.append(audio_files_nametrue[i][0])   
    random.shuffle(nametrue)
    nametrue40=nametrue[0:soundNum*totalrunnum]
audio_files_namefalse=[os.path.split(f)[1].split(".") for f in os.listdir("False") \
                       if os.path.split(f)[1].split(".")[1] == 'wav']
for i in range(len(audio_files_namefalse)):
    namefalse.append(audio_files_namefalse[i][0])    
    random.shuffle(namefalse)
    namefalse40=namefalse[0:soundNum*totalrunnum]
    

Time2=time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())) 
os.makedirs(r'%s/%s'%(current_dir,Time2))# creat a sub_file to save all of the files
root=Tk()
root.title('BCI experiment')
root.geometry('64x64')
root.iconbitmap('logooo.ico')
sh=root.winfo_screenheight()
sw=root.winfo_screenwidth() 
root.wm_state( 'zoomed' )
label1=Label(root,text='Brain Computer Interface for CLIS',fg='black',width=100,height=4,font=("Fixdsys",36,"bold"))
label1.pack()
label2=Label(root,text='Institute of Medical Psychology and Behavioral Neurobiology',fg='black',width=100,height=2,font=("Fixdsys",18,"bold"))
label2.pack(side='bottom')
bm=PhotoImage(file='logo0.gif')
label3=Label(root,image=bm)
label3.place(x=1200,y=7,anchor='nw')
#label3.pack(side='bottom')
#pic=Image.open("logo.jpg")
#pic=PhotoImage(file="logo.gif")
#Label(root,image=pic).pack()
#label2.place(x=1400,y=850)
cv=Canvas(root,width=sw,height=sh*0.4,bg='white')

#cv.creat_image(image=pic)
os.chdir(current_dir+'\\'+Time2)
filr_path=os.getcwd()
#cv=Canvas(root,bg='white')
#sel=cv.create_rectangle(10,10,50,50,outline='blue',fill='gray')
#cv.coords(cv,225,225,1440,1440)
rt=cv.create_text(sw/2,sh/6,text='',fill='black',font=("Fixdsys",36,"bold"))   
button=Button(root,text='calibration',fg='green',bg='blue',font=("Fixdsys",18,"bold"),command=buttoncallback)
cv.pack() 
button.pack(side='bottom')
root.mainloop()


