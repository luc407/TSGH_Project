# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 21:39:48 2022

@author: luc40
"""
import os
import cv2
import numpy as np
import Neurobit as nb
from Neurobit import Neurobit
from scipy import stats
from matplotlib import pyplot as plt
        
class CUT_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "CUT"
        self.sequence = 0
        self.FolderName = csv_path.split('\\')[-2]
        self.FileName = csv_path.split('\\')[-1].replace(".csv","")
        self.main_path = csv_path.replace("\\"+csv_path.split('\\')[-2],"").replace("\\"+csv_path.split('\\')[-1],"")
        self.DB_path = os.path.abspath(self.main_path+"\\../..")
        self.save_MainPath = self.save_path+"\\"+self.FolderName
        self.saveReport_path = self.save_MainPath
        self.saveMerge_path = self.save_MainPath+"\\"+self.task
        self.saveVideo_path = self.save_MainPath+"\\"+self.task+"\\HoughCircle"
        self.saveImage_path = self.save_MainPath+"\\"+self.task+"\\Image"              
        if not os.path.isdir(self.saveVideo_path):
            os.makedirs(self.saveVideo_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
        if not os.path.isdir(self.saveImage_path):
            os.makedirs(self.saveImage_path)
    def Exec(self):
        self.GetCommand()  
        
        self.GetEyePosition()  
        #self.SeperateSession()              
        self.FeatureExtraction()  
        self.GetDiagnosis()  
        self.Save2Cloud()
        
        self.DrawEyeFig()
        self.DrawEyeTrack()  
        self.DrawQRCode()
    def FeatureExtraction(self):
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        OD_ACT = []; OS_ACT = [];       # all position in each CUT_TIME
        for i in range(0,len(nb.CUT_TIME)):
            temp = self.CmdTime[nb.CUT_TIME[i]]
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)
            if type(self.CmdTime[nb.CUT_TIME[i]]) == list:
                OD_ACT_tmp = []; OS_ACT_tmp = [];
                for temp in self.CmdTime[nb.CUT_TIME[i]]:
                        OD_ACT_tmp.append(stats.mode(OD[:,temp].astype(int),axis = 1)[0].reshape(-1))
                        OS_ACT_tmp.append(stats.mode(OS[:,temp].astype(int),axis = 1)[0].reshape(-1))
                OD_ACT_tmp = np.array(OD_ACT_tmp)
                OS_ACT_tmp = np.array(OS_ACT_tmp)
                OD_ACT.append(np.round(stats.mode(OD_ACT_tmp.astype(int),axis = 1)[0][0].reshape(-1),2))
                OS_ACT.append(np.round(stats.mode(OS_ACT_tmp.astype(int),axis = 1)[0][0].reshape(-1),2))
            elif temp.any():                
                OD_ACT.append(stats.mode(OD[:,temp].astype(int),axis = 1)[0].reshape(-1))
                OS_ACT.append(stats.mode(OS[:,temp].astype(int),axis = 1)[0].reshape(-1))
            else:
                OD_ACT.append([np.nan, np.nan, np.nan])
                OS_ACT.append([np.nan, np.nan, np.nan])
        
        # ET、XT angle
        OD_ACT = np.array(np.round(OD_ACT,2))
        OS_ACT = np.array(np.round(OS_ACT,2))
        self.OD_ACT = OD_ACT
        self.OS_ACT = OS_ACT
        # Fixation_eye - Uncovered_eye
        OS_phoria = OS_ACT[2]-OS_ACT[1]    # UCL-CL
        OD_phoria = OD_ACT[4]-OD_ACT[3]    # UCR-CR
        
        OD_fix = OD_ACT[1]-OD_ACT[3]    # CL-CR
        OS_fix = OS_ACT[3]-OS_ACT[1]    # CR-CL
        
        plt.subplot(2,2,1)
        plt.plot(self.OD[0,:]-min(self.OD[0,:]))
        plt.plot(self.OD[1,:]-min(self.OD[1,:]))
        plt.hlines(OD_ACT[2][0]-min(self.OD[0,:]),0,len(self.OD[1,:]), color ='r')# CL
        plt.hlines(OD_ACT[2][1]-min(self.OD[1,:]),0,len(self.OD[1,:]), color ='g')# CL
        plt.hlines(OD_ACT[1][0]-min(self.OD[0,:]),0,len(self.OD[1,:]), color ='r')# CR
        plt.hlines(OD_ACT[1][1]-min(self.OD[1,:]),0,len(self.OD[1,:]), color ='g')# CR
        plt.subplot(2,2,2)
        plt.plot(self.OS[0,:]-min(self.OS[0,:]))
        plt.plot(self.OS[1,:]-min(self.OS[1,:]))
        plt.hlines(OS_ACT[2][0]-min(self.OS[0,:]),0,len(self.OS[1,:]), color ='r')# CL
        plt.hlines(OS_ACT[2][1]-min(self.OS[1,:]),0,len(self.OS[1,:]), color ='g')# CL
        plt.hlines(OS_ACT[1][0]-min(self.OS[0,:]),0,len(self.OS[1,:]), color ='r')# CR
        plt.hlines(OS_ACT[1][1]-min(self.OS[1,:]),0,len(self.OS[1,:]), color ='g')# CR
        plt.subplot(2,2,3)
        plt.plot(self.OD[0,:]-min(self.OD[0,:]))
        plt.plot(self.OD[1,:]-min(self.OD[1,:]))
        plt.hlines(OD_ACT[4][0]-min(self.OD[0,:]),0,len(self.OD[1,:]), color ='r')# UCL
        plt.hlines(OD_ACT[4][1]-min(self.OD[1,:]),0,len(self.OD[1,:]), color ='g')# UCR
        plt.hlines(OD_ACT[3][0]-min(self.OD[0,:]),0,len(self.OD[1,:]), color ='r')# UCL
        plt.hlines(OD_ACT[3][1]-min(self.OD[1,:]),0,len(self.OD[1,:]), color ='g')# UCR
        plt.subplot(2,2,4)
        plt.plot(self.OS[0,:]-min(self.OS[0,:]))
        plt.plot(self.OS[1,:]-min(self.OS[1,:]))
        plt.hlines(OS_ACT[4][0]-min(self.OS[0,:]),0,len(self.OS[1,:]), color ='r')# UCL
        plt.hlines(OS_ACT[4][1]-min(self.OS[1,:]),0,len(self.OS[1,:]), color ='g')# UCR
        plt.hlines(OS_ACT[3][0]-min(self.OS[0,:]),0,len(self.OS[1,:]), color ='r')# UCL
        plt.hlines(OS_ACT[3][1]-min(self.OS[1,:]),0,len(self.OS[1,:]), color ='g')# UCR
        plt.show()
        
        try:
            OD_fix = np.append(nb.trans_PD(self.AL_OD,OD_fix[0:2],nb.CAL_VAL_OD), OD_fix[2])
            OS_fix = np.append(nb.trans_PD(self.AL_OS,OS_fix[0:2],nb.CAL_VAL_OS), OS_fix[2])
            
            OS_phoria = np.append(nb.trans_PD(self.AL_OS,OS_phoria[0:2],nb.CAL_VAL_OS), OS_phoria[2])
            OD_phoria = np.append(nb.trans_PD(self.AL_OD,OD_phoria[0:2],nb.CAL_VAL_OD), OD_phoria[2])
        except:
            print("No profile")
        self.OD_fix = OD_fix        # one position in each CUT_TIME
        self.OS_fix = OS_fix
        self.OD_phoria = OD_phoria        # one position in each CUT_TIME
        self.OS_phoria = OS_phoria
    def GetDiagnosis(self):
        OD_fix = self.OD_fix; OS_fix = self.OS_fix
        OD_phoria = self.OD_phoria; OS_phoria = self.OS_phoria
        thr =1.5
        self.NeurobitDx_H = None
        self.NeurobitDx_V = None
        self.NeurobitDxTp_X = None
        if np.all(np.abs([OD_fix,OS_fix,OS_phoria,OD_phoria])<=thr):
            self.Ortho = True
            self.NeurobitDx_H = 'Ortho'
            self.NeurobitDx_V = 'Ortho'
            self.NeurobitDxTp_H = 'None'
            self.NeurobitDxDev_H = 0
            self.NeurobitDxDev_V = 0
        else:
            self.Ortho = False
        """Tropia"""
        if -OS_fix[0]>thr or OD_fix[0]>thr:
            self.NeurobitDx_H = 'XT'
            if -OS_fix[0]>thr and OD_fix[0]>thr:
                self.NeurobitDxTp_H = 'Divergence'
                self.NeurobitDxDev_H = (abs(OD_fix[0])+abs(OS_fix[0]))/2
            elif -OS_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OS_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OS, Levoversion'
                else:
                    self.NeurobitDxTp_X = 'OS, Divergence'
            elif OD_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OD_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OD, Levoversion'
                else:
                    self.NeurobitDxTp_X = 'OD, Divergence'            
        elif OS_fix[0]>thr or -OD_fix[0]>thr:
            self.NeurobitDx_H = 'ET'
            if OS_fix[0]>thr and -OD_fix[0]>thr:
                self.NeurobitDxTp_H = 'Convergence'
                self.NeurobitDxDev_H = (abs(OD_fix[0])+abs(OS_fix[0]))/2
            elif OS_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OS_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OS Detroversion'
                else:
                    self.NeurobitDxTp_X = 'OS Convergence'           
            elif -OD_fix[0]>thr:
                self.NeurobitDxDev_H = abs(OD_fix[0])
                if OS_fix[0]*OD_fix[0]>0:
                    self.NeurobitDxTp_X = 'OD Detroversion'
                else:
                    self.NeurobitDxTp_X = 'OD Convergence'
        else:
            self.NeurobitDx_H = 'Ortho'
            self.NeurobitDxDev_H = 0
        
        """Phoria"""
        if OS_phoria[0]>thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OS XP"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OS_phoria[0]))
        elif OS_phoria[0]<-thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OS E"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OS_phoria[0]))
        else: self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H,0)
            
        if OD_phoria[0]>thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OD XP"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OD_phoria[0]))
        elif OD_phoria[0]<-thr:
            self.NeurobitDx_H = self.NeurobitDx_H +" & "+ "OD E"
            self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H, abs(OD_phoria[0]))
        else: self.NeurobitDxDev_H = np.append(self.NeurobitDxDev_H,0)
        
        """Vertical"""
        if OS_fix[1]>thr or -OD_fix[1]>thr:
            self.NeurobitDx_V = 'LHT'
            if OS_fix[1]>thr:
                self.NeurobitDxDev_V = abs(OS_fix[1])
            else:
                self.NeurobitDxDev_V = abs(OD_fix[1])
        
        elif -OS_fix[1]>thr or OD_fix[1]>thr:
            self.NeurobitDx_V = 'LHoT'
            if OS_fix[1]>thr:
                self.NeurobitDxDev_V = abs(OS_fix[1])
            else:
                self.NeurobitDxDev_V = abs(OD_fix[1])
        else:
            self.NeurobitDx_V = 'Ortho'
            self.NeurobitDxDev_V = 0    
        
        if OD_phoria[1]>thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OD HP"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OD_phoria[1]))
        elif OD_phoria[1]<-thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OD H"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OD_phoria[1]))
        else: self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V,0)
        
        if OS_phoria[1]>thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OS HP"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OS_phoria[1]))
        elif OS_phoria[1]<-thr:
            self.NeurobitDx_V = self.NeurobitDx_V +" & "+ "OS H"
            self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V, abs(OS_phoria[1]))
        else: self.NeurobitDxDev_V = np.append(self.NeurobitDxDev_V,0)
        
        nb.CUT_Save._CUT_dx['ID'].append(self.ID)
        nb.CUT_Save._CUT_dx['H_Dx'].append(self.NeurobitDx_H)
        nb.CUT_Save._CUT_dx['H_Dev'].append(self.NeurobitDxDev_H)
        nb.CUT_Save._CUT_dx['H_type'].append(self.NeurobitDxTp_X)
        nb.CUT_Save._CUT_dx['V_Dx'].append(self.NeurobitDx_V)
        nb.CUT_Save._CUT_dx['V_Dev'].append(self.NeurobitDxDev_V)
    def GetTimeFromCmd(self):
        cmd = self.VoiceCommand
        O_t = np.where(cmd==0)[0]
        CL_t = np.where(cmd==1)[0]
        UCL_t = np.where(cmd==2)[0]
        CR_t = np.where(cmd==3)[0] 
        UCR_t = np.where(cmd==4)[0]
        self.CmdTime = {"CL_t": np.array(CL_t),
                        "UCL_t": np.array(UCL_t),
                        "CR_t": np.array(CR_t),
                        "UCR_t": np.array(UCR_t),
                        "O_t":  np.array(O_t)}
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(nb.CUT_TIME)):
            temp = np.array(self.CmdTime[nb.CUT_TIME[i]])
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)
            diff_temp = np.diff(temp)
            inds = np.where(diff_temp>20)[0]
            if len(inds)>0:
                list_temp = list(); j = 0
                for ind in inds:
                    list_temp.append(temp[j:ind])
                    j = ind
                list_temp.append(temp[ind:])
                self.CmdTime[nb.CUT_TIME[i]] = list_temp
            else:
                self.CmdTime[nb.CUT_TIME[i]] = temp
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/25
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(nb.EYE)):
            if nb.EYE[i] == 'OD':
                x_diff = self.OD_ACT[0,0]-OD[0,:]
                y_diff = self.OD_ACT[0,1]-OD[1,:]
                x_PD = nb.trans_PD(self.AL_OD,x_diff,nb.CAL_VAL_OD)
                y_PD = nb.trans_PD(self.AL_OD,y_diff,nb.CAL_VAL_OD)
            else:
                x_diff = self.OS_ACT[0,0]-OS[0,:]
                y_diff = self.OS_ACT[0,1]-OS[1,:]
                x_PD = nb.trans_PD(self.AL_OS,x_diff,nb.CAL_VAL_OS)
                y_PD = nb.trans_PD(self.AL_OS,y_diff,nb.CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position (PD)")
            plt.title("Cover Uncover Test "+ nb.EYE[i])
            
            plt.grid(True, linestyle=':')
            plt.xticks(fontsize= 8)
            plt.yticks(fontsize= 8)
            
            plt.text(0,90, "right",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(0,-90, "left",color='lightsteelblue' ,
                     horizontalalignment='left',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1], 90,"up",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8)
            plt.text(time[-1], -90,"down",color='salmon',
                     horizontalalignment='right',
                     verticalalignment='center', fontsize=8) 
            plt.ylim([-100,100])
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300) 
    def DrawEyeFig(self):
        ACT = []; OD = self.OD; OS = self.OS
        for i in range(0,len(self.OS_ACT)):
            try:t = np.concatenate(np.array(self.CmdTime[nb.CUT_TIME[i]]))
            except:t = self.CmdTime[nb.CUT_TIME[i]]
            if not np.isnan(self.OD_ACT[i,0]) and not np.isnan(self.OS_ACT[i,0]):
                OD_diff = abs(OD[0,t]-self.OD_ACT[i,0])+abs(OD[1,t]-self.OD_ACT[i,1])
                OS_diff = abs(OS[0,t]-self.OS_ACT[i,0])+abs(OS[1,t]-self.OS_ACT[i,1])
                Diff = np.sum(np.array([OD_diff, OS_diff]),axis = 0)
                pupil = OS[2,t]+OD[2,t]
            elif np.isnan(self.OD_ACT[i,0]):
                Diff = abs(OS[0,t]-self.OS_ACT[i,0])+abs(OS[1,t]-self.OS_ACT[i,1])
                pupil = OS[2,t]
            else:
                Diff = abs(OD[0,t]-self.OD_ACT[i,0])+abs(OD[1,t]-self.OD_ACT[i,1])
                pupil = OD[2,t]
            try:
# =============================================================================
#                 ind = np.where(Diff == np.nanmin(Diff))[0]
#                 ind_pu = np.where(pupil[ind] == np.nanmax(pupil[ind]))[0]
# =============================================================================
                ACT.append(t[np.where(Diff == np.nanmin(Diff))[0][0]])
            except:
                ACT.append(ACT[-1])
                #print("Not Detect "+ CUT_TIME[i]) 
        pic_cont = 1
        empt=0
        #fig = plt.figure(figsize=(11.7,8.3))
        fig = plt.gcf()
        fig.set_size_inches(3,5, forward=True)
        fig.set_dpi(300)
        for pic in ACT:
            cap = nb.GetVideo(self.csv_path)
            cap.set(1,pic)
            ret, im = cap.read()
            height = im.shape[0]
            width = im.shape[1]
            try:
                cv2.rectangle(im,
                              (int(self.OD_ACT[pic_cont-1][0]),int(self.OD_ACT[pic_cont-1][1])),
                              (int(self.OD_ACT[pic_cont-1][0])+1,int(self.OD_ACT[pic_cont-1][1])+1),
                              (0,255,0),2)
                cv2.circle(im,(int(self.OD_ACT[pic_cont-1][0]),int(self.OD_ACT[pic_cont-1][1])),
                           int(self.OD_ACT[pic_cont-1][2]),
                           (255,255,255),2) 
            except:
                pass#print("OD Absent!")
            try:
                cv2.rectangle(im,
                              (int(self.OS_ACT[pic_cont-1][0]),int(self.OS_ACT[pic_cont-1][1])),
                              (int(self.OS_ACT[pic_cont-1][0])+1,int(self.OS_ACT[pic_cont-1][1])+1),
                              (0,255,0),2)
                cv2.circle(im,(int(self.OS_ACT[pic_cont-1][0]),int(self.OS_ACT[pic_cont-1][1])),
                           int(self.OS_ACT[pic_cont-1][2]),
                           (255,255,255),2)
            except:
                pass#print("OS Absent!")
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            _,thresh_1 = cv2.threshold(gray,110,255,cv2.THRESH_TRUNC)
            exec('ax'+str(pic_cont)+'=plt.subplot(5, 1, pic_cont)')
            exec('ax'+str(pic_cont)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
            exec('ax'+str(pic_cont)+'.axes.xaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.axes.yaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.set_ylim(int(3*height/4),int(height/4))')
# =============================================================================
#             exec('ax'+str(pic_cont)+ '.set_ylim(int(height),int(0))')
# =============================================================================
            exec('ax'+str(pic_cont)+ '.set_ylabel(nb.CUT_LABEL[pic_cont-1])')
            plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]
        for i in range(0,len(nb.CUT_TIME)):
            if frame_cnt in self.CmdTime[nb.CUT_TIME[i]]:
                text = nb.CUT_STR[i]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)[0]
                textX = int((width - textsize[0]) / 2)
                cv2.putText(frame,text, (textX, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 
                            2, (255, 255, 255),
                            2, cv2.LINE_AA)
        if self.IsVoiceCommand:
            text = "Voice Command"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)[0]
            textX = int((width - textsize[0]) / 2)
            cv2.putText(frame,text, (textX, 550), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (0, 255, 255),
                        1, cv2.LINE_AA)
        else:
            text = "No Voice Command"
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)[0]
            textX = int((width - textsize[0]) / 2)
            cv2.putText(frame,text, (textX, 550), 
                        cv2.FONT_HERSHEY_TRIPLEX, 
                        1, (0, 255, 255),
                        1, cv2.LINE_AA)
    