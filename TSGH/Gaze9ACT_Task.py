# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 18:02:16 2023

@author: luc40
"""

import os
import cv2
import math
import tkinter as tk
import numpy as np
import Neurobit as nb
from Neurobit import Neurobit
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from EyeTrack import EyeTrackSys
    
class Gaze9ACT_Task(Neurobit):
    def __init__(self, csv_path):
        nb.Neurobit.__init__(self)
        self.task = "9_GazeACT"
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
    def Exec(self,*args):
        Finished_ACT = False
        for arg in args:
            if arg: 
                Finished_ACT = True
                ACT_Task = arg
        self.GetCommand()   
        self.GetEyePosition()
        self.Save2Cloud()
        self.DrawQRCode() 
        
# =============================================================================
#         self.Preprocessing()
#         #self.SeperateSession()                
#         if Finished_ACT: self.FeatureExtraction(ACT_Task) 
#         else: self.FeatureExtraction() 
#         self.DrawEyeFig()
#         self.DrawEyeMesh()
#         self.DrawEyeTrack()  
# =============================================================================
    def GetTimeFromCmd(self):        
        x = np.round(self.VoiceCommand[0,:],2); y = np.round(self.VoiceCommand[1,:],2); self.CmdTime = dict()
        cover = self.VoiceCommand[2,:]
        
        x_q1 = np.unique(x)[0]
        x_q2 = np.unique(x)[1]
        x_q3 = -np.unique(x)[0]

        y_q1 = np.unique(y)[0]
        y_q2 = np.unique(y)[1]
        y_q3 = -np.unique(y)[0]
        
        D = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), y == y_q1))[0]#[60:]
        F = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), np.logical_and(y < y_q3, y > y_q1)))[0]#[60:]
        L = np.where(np.logical_and(np.logical_and(x == x_q1, y < y_q3), y > y_q1))[0]#[60:]
        LD = np.where(np.logical_and(x == x_q1, y == y_q1))[0]#[60:]
        LU = np.where(np.logical_and(x == x_q1, y == y_q3))[0]#[60:]
        R = np.where(np.logical_and(np.logical_and(x == x_q3 ,  y < y_q3), y > y_q1))[0]#[60:]
        RD = np.where(np.logical_and(x == x_q3, y == y_q1))[0]#[60:]
        RU = np.where(np.logical_and(x == x_q3, y == y_q3))[0]#[60:]
        U = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), y == y_q3))[0]#[60:]
        
        for i in range(0,len(nb.GAZE_9_TIME)):
            exec('cov_non = np.where(cover['+nb.GAZE_9_TIME[i]+'] >= 0)[0]')
            exec('self.CmdTime[nb.GAZE_9_TIME[i]] = '+nb.GAZE_9_TIME[i]+'[cov_non]')
    def FeatureExtraction(self):
        
        def GetTrialCmd(i,eyePosition,Gaze9,T,CmdTime,AdjCmdTime):
            window = int(4*24) # respond period = seconds*fps 
            LT = 20  # Set default latency
            CmdTmp = CmdTime[T];#print(T,i)       
            CmdTmp = CmdTmp[CmdTmp<eyePosition.shape[1]-1]
            
            Trial_trg_ind = np.where(np.diff(CmdTime[T]) > 5)[0]
            start_ind = np.where(CmdTmp == i)[0][0]
            end_ind = Trial_trg_ind[np.where(Trial_trg_ind>start_ind)[0]]
            if end_ind.any(): end_ind = end_ind[0]
            else: end_ind = len(CmdTmp)-1
            
            
            
            t = CmdTmp[start_ind]-LT;            
            baseline_x = np.nanmean(eyePosition[0,CmdTmp[start_ind]-window:CmdTmp[start_ind]-window+LT]) # mean value in latency
            baseline_y = np.nanmean(eyePosition[1,CmdTmp[start_ind]-window:CmdTmp[start_ind]-window+LT]) # mean value in latency
            
            tmp_t = []; 
            if T == 'F':                
                try:
                    Gaze9[T] = np.append(Gaze9[T], [np.nanpercentile(eyePosition[:,CmdTmp[start_ind:end_ind]],50,axis = 1)],axis = 0)
                except:
                    Gaze9[T] = [np.nanpercentile(eyePosition[:,CmdTmp[start_ind:end_ind]],50,axis = 1)]
                try:
                    AdjCmdTime[T] = np.append(AdjCmdTime[T],CmdTmp[end_ind])
                except:
                    AdjCmdTime[T] = CmdTmp[end_ind]
            elif T == 'U':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if y < pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'D':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if y > pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'R':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x < pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'L':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x > pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'RU':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x < pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'LU':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x > pre_x or y < pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'RD':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x < pre_x:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            elif T == 'LD':
                pre_x = baseline_x; pre_y = baseline_y;
                for x,y,p in eyePosition[:,CmdTmp[start_ind]-LT:CmdTmp[end_ind]+window].transpose():
                    t+=1
                    if x > pre_x or y > pre_y:
                        fin_x = x; fin_y = y; fin_p = p;
                        pre_x = x; pre_y = y;
                        tmp_t = t
            if not tmp_t: 
                tmp_t = CmdTmp[start_ind]
                fin_x, fin_y, fin_p = list(np.nanmean(eyePosition[:,CmdTmp[start_ind:end_ind]],axis = 1))
            if T != 'F':
                try:
                    Gaze9[T] = np.append(Gaze9[T], [[fin_x,fin_y,fin_p]],axis = 0)
                except:
                    Gaze9[T] = [[fin_x,fin_y,fin_p]]
                try:
                    AdjCmdTime[T] = np.append(AdjCmdTime[T],tmp_t)
                except:
                    AdjCmdTime[T] = tmp_t
            i = CmdTmp[end_ind]+1
            return Gaze9, AdjCmdTime, i
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        
        # delete bias        
        for i in range(0,len(nb.GAZE_9_TIME)):
            temp = np.array(self.CmdTime[nb.GAZE_9_TIME[i]])
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                self.CmdTime[nb.GAZE_9_TIME[i]] = np.delete(temp, delete)
                
        AdjCmdTime = dict()
        OD_Gaze9 = dict()
        OS_Gaze9 = dict()
                
        i = 0; PASS = False;      # read CmdTime        
        while(i < len(OD[0,:])):
            if i == len(OD[0,:])-1: break
            for TIME in range(0,len(nb.GAZE_9_TIME)):                
                if i in self.CmdTime[nb.GAZE_9_TIME[TIME]]:
                    T = nb.GAZE_9_TIME[TIME]
                    OD_Gaze9,AdjCmdTime,_ = GetTrialCmd(i,OD,OD_Gaze9,T,self.CmdTime,AdjCmdTime)
                    OS_Gaze9,AdjCmdTime,i = GetTrialCmd(i,OS,OS_Gaze9,T,self.CmdTime,AdjCmdTime)
                    PASS = True
                    break
                else:
                    PASS = False                
            if not PASS:
                i+=1
             
        #GAZE_9_TIME     = ['F','U','D','R','L','RU','LU','RD','LD']
        """Get O, CL, CR's OD and OS movement:
            self.NeurobitDxDev_H[cmd] = [O: OD_x, OS_x,
                                         CL:OD_x, OS_x,
                                         CR:OD_x, OS_x]
            self.NeurobitDxDev_V[cmd] = [O: OD_y, OS_y,
                                         CL:OD_y, OS_y,
                                         CR:OD_y, OS_y]
        """
        self.NeurobitDxDev_H = dict();self.NeurobitDxDev_V = dict()
        self.Gaze_9_OD = OD_Gaze9
        self.Gaze_9_OS = OS_Gaze9
        self.CmdTime = AdjCmdTime
                    
        for cmd in nb.GAZE_9_TIME:
            diff_OD_O = self.Gaze_9_OD['F'][1]-self.Gaze_9_OD[cmd][0]
            diff_OD_CL = self.Gaze_9_OD['F'][1]-self.Gaze_9_OD[cmd][1]
            diff_OD_CR = self.Gaze_9_OD['F'][1]-self.Gaze_9_OD[cmd][2]
            
            diff_OS_O = self.Gaze_9_OS['F'][2]-self.Gaze_9_OS[cmd][0]
            diff_OS_CL = self.Gaze_9_OS['F'][2]-self.Gaze_9_OS[cmd][1]
            diff_OS_CR = self.Gaze_9_OS['F'][2]-self.Gaze_9_OS[cmd][2]
            
            PD_OD_O = nb.trans_AG(self.AL_OD,np.array(diff_OD_O[:2]),nb.CAL_VAL_OD)
            PD_OD_CL = nb.trans_AG(self.AL_OD,np.array(diff_OD_CL[:2]),nb.CAL_VAL_OD)
            PD_OD_CR = nb.trans_AG(self.AL_OD,np.array(diff_OD_CR[:2]),nb.CAL_VAL_OD)
            
            PD_OS_O = nb.trans_AG(self.AL_OS,np.array(diff_OS_O[:2]),nb.CAL_VAL_OS)
            PD_OS_CL = nb.trans_AG(self.AL_OS,np.array(diff_OS_CL[:2]),nb.CAL_VAL_OS)
            PD_OS_CR = nb.trans_AG(self.AL_OS,np.array(diff_OS_CR[:2]),nb.CAL_VAL_OS)
            
            self.NeurobitDxDev_H[cmd] = np.round([[PD_OD_O[0],PD_OS_O[0]], [PD_OD_CL[0],PD_OS_CL[0]], [PD_OD_CR[0],PD_OS_CR[0]]],2)
            self.NeurobitDxDev_V[cmd] = np.round([[PD_OD_O[1],PD_OS_O[1]], [PD_OD_CL[1],PD_OS_CL[1]], [PD_OD_CR[1],PD_OS_CR[1]]],2)
            
            if len(self.NeurobitDxDev_H[cmd])==3:
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OD_H_Dev'].append(self.NeurobitDxDev_H[cmd].transpose()[0])
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OS_H_Dev'].append(self.NeurobitDxDev_H[cmd].transpose()[1])
            else:
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OD_H_Dev'].append([np.nan, np.nan, np.nan])
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OS_H_Dev'].append([np.nan, np.nan, np.nan])
            if len(self.NeurobitDxDev_V[cmd])==3:
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OD_V_Dev'].append(self.NeurobitDxDev_V[cmd].transpose()[0])
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OS_V_Dev'].append(self.NeurobitDxDev_V[cmd].transpose()[1])
            else:
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OD_V_Dev'].append([np.nan, np.nan, np.nan])
                nb.Gaze9_Save._Gaze9_dx[cmd+'_OS_V_Dev'].append([np.nan, np.nan, np.nan])
        """Get O, CL, CR's OD and OS movement area:
            nb.Gaze9_Save._Gaze9_dx['OD_Area'] = [O,CL,CR]
            nb.Gaze9_Save._Gaze9_dx['OS_Area'] = [O,CL,CR]
        """
        a = [];b = []; OD_Area = []
        for i in range(0,3):
            for cmd in nb.GAZE_9_BOARDER:
                a.append(self.NeurobitDxDev_H[cmd].transpose()[0][i])                
                b.append(self.NeurobitDxDev_V[cmd].transpose()[0][i])            
            xy = np.array([a,b]).transpose()
            OD_Area.append(np.round(nb.enclosed_area(xy),2))
        nb.Gaze9_Save._Gaze9_dx['OD_Area'].append(OD_Area)
       
        a = [];b = []; OS_Area =  []
        for i in range(0,3):
           for cmd in nb.GAZE_9_BOARDER:
               a.append(self.NeurobitDxDev_H[cmd].transpose()[1][i])                
               b.append(self.NeurobitDxDev_V[cmd].transpose()[1][i])            
           xy = np.array([a,b]).transpose()
           OS_Area.append(np.round(nb.enclosed_area(xy),2))
        nb.Gaze9_Save._Gaze9_dx['OS_Area'].append(OS_Area)
        
        self.SaveDxTrue(nb.Gaze9_Save)
        nb.Gaze9_Save._Gaze9_dx['ID'].append(self.ID)
        nb.Gaze9_Save._Gaze9_dx['Examine Date'].append(self.FolderName.split('_')[0])
        
        
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(nb.GAZE_9_TIME)):
            temp = self.CmdTime[nb.GAZE_9_TIME[i]]
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
                self.CmdTime[nb.GAZE_9_TIME[i]] = list_temp
            else:
                self.CmdTime[nb.GAZE_9_TIME[i]] = temp
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]      
        for i in range(0,len(nb.GAZE_9_TIME)):
            if frame_cnt in self.CmdTime[nb.GAZE_9_TIME[i]]:
                text = nb.GAZE_9_STR[i]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)[0]
                textX = int((width - textsize[0]) / 2)
                cv2.putText(frame,text, (textX, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 
                            2, (255, 255, 255),
                            2, cv2.LINE_AA)        
    def DrawEyeTrack(self):
        for ss in range(0,3):
            OD = self.OD; OS = self.OS
            keep = np.where(self.VoiceCommand[2,:]==ss)[0]
            delete = np.where(keep>len(OD[0])-1)[0]
            if delete.any():
                keep = np.delete(keep, delete)
            OD = self.OD[:,keep]; OS = self.OS[:,keep]
            time = np.array(range(0,len(OD[0])))/25
            fig = plt.gcf()
            fig.set_size_inches(7.2,2.5, forward=True)
            fig.set_dpi(300)              
            for i in range(0,len(nb.EYE)):
                if nb.EYE[i] == 'OD':
                    x_diff = self.Gaze_9_OD['F'][1][0]-OD[0,:]
                    y_diff = self.Gaze_9_OD['F'][1][1]-OD[1,:]
                    x_PD = nb.trans_AG(self.AL_OD,x_diff,nb.CAL_VAL_OD)
                    y_PD = nb.trans_AG(self.AL_OD,y_diff,nb.CAL_VAL_OD)
                else:
                    x_diff = self.Gaze_9_OS['F'][2][0]-OS[0,:]
                    y_diff = self.Gaze_9_OS['F'][2][1]-OS[1,:]
                    x_PD = nb.trans_AG(self.AL_OS,x_diff,nb.CAL_VAL_OS)
                    y_PD = nb.trans_AG(self.AL_OS,y_diff,nb.CAL_VAL_OS)
                plt.subplot(1,2,i+1)
                plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
                plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
                plt.xlabel("Time (s)")
                plt.ylabel("Eye Position ("+chr(176)+")")
                plt.title("9 Gaze Test "+ nb.EYE[i])
                plt.grid(True, linestyle=':')
                plt.xticks(fontsize= 8)
                plt.yticks(fontsize= 8)
                plt.ylim([-100,100])
                plt.text(0,90, "right",color='lightsteelblue' ,
                         horizontalalignment='left',
                         verticalalignment='center', fontsize=8)
                plt.text(0,-90, "left",color='lightsteelblue' ,
                         horizontalalignment='left',
                         verticalalignment='center', fontsize=8)
                plt.text(time[-1],90,"up",color='salmon',
                         horizontalalignment='right',
                         verticalalignment='center', fontsize=8)
                plt.text(time[-1], -90,"down",color='salmon',
                         horizontalalignment='right',
                         verticalalignment='center', fontsize=8)        
            plt.tight_layout()
            plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack_"+nb.ACT_LABEL[ss]+".png"), dpi=300) 
            plt.close(fig)
            plt.clf()
    def DrawEyeFig(self):
        for ss in range(0,3):
            OD = self.OD; OS = self.OS
            Gaze_9 = []  
            for cmd in nb.GAZE_9_TIME:
                Gaze_9.append(self.CmdTime[cmd][2*ss])
# =============================================================================
#                 try:t = np.concatenate(np.array(self.CmdTime[cmd]))
#                 except:t = self.CmdTime[cmd]
#     
#                 delete = np.where(t>len(OD[0])-1)[0]
#                 if delete.any():
#                     t = np.delete(t, delete)
#                     
#                 if not np.isnan(self.Gaze_9_OD[cmd][ss]).any() and not np.isnan(self.Gaze_9_OS[cmd][ss]).any():
#                     OD_diff = abs(OD[0,t]-self.Gaze_9_OD[cmd][ss][0])+abs(OD[1,t]-self.Gaze_9_OD[cmd][ss][1])
#                     OS_diff = abs(OS[0,t]-self.Gaze_9_OS[cmd][ss][0])+abs(OS[1,t]-self.Gaze_9_OS[cmd][ss][1])
#                     Diff = []
#                     for x in np.array([OD_diff,OS_diff]).T:
#                         if not np.all(np.isnan(x)):
#                             Diff.append(np.nansum(x))
#                         else:
#                             Diff.append(np.nan)
#                     Diff = np.array(Diff)
#                 elif np.isnan(self.Gaze_9_OD[cmd][ss]).any():
#                     Diff = abs(OS[0,t]-self.Gaze_9_OS[cmd][ss][0])+abs(OS[1,t]-self.Gaze_9_OS[cmd][ss][1])
#                 else:
#                     Diff = abs(OD[0,t]-self.Gaze_9_OD[cmd][ss][0])+abs(OD[1,t]-self.Gaze_9_OD[cmd][ss][1])              
#                 
#                 if not np.isnan(Diff).all():
#                     Gaze_9.append(t[np.where(Diff == np.nanmin(Diff))[0][0]])
#                 else:
#                     Gaze_9.append(np.nan)
# =============================================================================
            pic_cont = 0
            empt=0
            fig = plt.gcf()
            fig.set_size_inches(7.2,2.5, forward=True)
            fig.set_dpi(300)
            for i in range(0,9):
                pic = Gaze_9[i]
                cmd = nb.GAZE_9_TIME[i]
                if not np.isnan(pic):
                    cap = nb.GetVideo(self.csv_path)
                    cap.set(1,pic)
                    ret, im = cap.read()
                    height = im.shape[0]
                    width = im.shape[1]
                    try:
                        cv2.rectangle(im,
                                      (int(self.Gaze_9_OD[cmd][ss][0]),int(self.Gaze_9_OD[cmd][ss][1])),
                                      (int(self.Gaze_9_OD[cmd][ss][0])+1,int(self.Gaze_9_OD[cmd][ss][1])+1),
                                      (0,255,0),2)
                        cv2.circle(im,(int(self.Gaze_9_OD[cmd][ss][0]),int(self.Gaze_9_OD[cmd][ss][1])),
                                   int(self.Gaze_9_OD[cmd][ss][2]),
                                   (255,255,255),2) 
                    except:
                        pass#print("OD Absent!")
                    try:
                        cv2.rectangle(im,
                                      (int(self.Gaze_9_OS[cmd][ss][0]),int(self.Gaze_9_OS[cmd][ss][1])),
                                      (int(self.Gaze_9_OS[cmd][ss][0])+1,int(self.Gaze_9_OS[cmd][ss][1])+1),
                                      (0,255,0),2)
                        cv2.circle(im,(int(self.Gaze_9_OS[cmd][ss][0]),int(self.Gaze_9_OS[cmd][ss][1])),
                                   int(self.Gaze_9_OS[cmd][ss][2]),
                                   (255,255,255),2)
                    except:
                        pass#print("OS Absent!")
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    _,thresh_1 = cv2.threshold(gray,110,255,cv2.THRESH_TRUNC)
                    
                    exec('ax'+str(pic_cont+1)+'=plt.subplot(3, 3, nb.GAZE_9_EYEFIG[pic_cont])')
                    exec('ax'+str(pic_cont+1)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
                    exec('ax'+str(pic_cont+1)+'.axes.xaxis.set_ticks([])')
                    exec('ax'+str(pic_cont+1)+ '.axes.yaxis.set_ticks([])')
                    exec('ax'+str(pic_cont+1)+ '.set_ylim(int(3*height/4),int(height/4))')
    # =============================================================================
    #                 exec('ax'+str(pic_cont+1)+ '.set_ylim(int(height),int(0))')
    # =============================================================================
                    exec('ax'+str(pic_cont+1)+ '.set_ylabel(nb.GAZE_9_TIME[pic_cont])')
                    plt.box(on=None)
                pic_cont+=1
            plt.tight_layout()
            plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig_"+nb.ACT_LABEL[ss]+".png"), dpi=300)
            plt.close(fig)
            plt.clf()
    def DrawEyeMesh(self):
        for ss in range(0,3):
            border = nb.GAZE_9_BOARDER
            NeurobitDxDev_H = []; NeurobitDxDev_V = []
            for cmd in border:
                NeurobitDxDev_H.append(self.NeurobitDxDev_H[cmd][ss])
                NeurobitDxDev_V.append(self.NeurobitDxDev_V[cmd][ss])
            NeurobitDxDev_H.append(self.NeurobitDxDev_H['F'][ss])
            NeurobitDxDev_V.append(self.NeurobitDxDev_V['F'][ss])
            NeurobitDxDev_H = np.array(NeurobitDxDev_H)
            NeurobitDxDev_V = np.array(NeurobitDxDev_V)
            MIN = -50; MAX = 50
            fig = plt.gcf()
            fig.set_size_inches(6/1.2,3/1.2, forward=True)
            fig.set_dpi(300)
            cir_size = 25
            try:
                diff_V = np.round(math.degrees(math.atan(abs(170-int(self.Height))/220)),2)
                if self.Height>170:
                    diff_V = -diff_V
            except:
                diff_V = 0
            for i in range(0,len(nb.EYE)):
                ax = plt.subplot(1,2,i+1)
                ax.xaxis.set_ticks(np.array(range(MIN,MAX,5)))
                ax.yaxis.set_ticks(np.array(range(MIN,MAX,5)))
                ax.grid(which='major') 
                ax.grid(which='minor') 
                majorLocator = MultipleLocator(25)
                minorLocator = MultipleLocator(5)
                ax.xaxis.set_major_locator(majorLocator)
                ax.xaxis.set_minor_locator(minorLocator)
                ax.yaxis.set_major_locator(majorLocator)
                ax.yaxis.set_minor_locator(minorLocator)
    
                plt.vlines(0,MIN,MAX,linewidth = .5,colors = 'k')
                plt.hlines(0,MIN,MAX,linewidth = .5,colors = 'k')
                plt.vlines(20,-15,15,linewidth = .5,colors = 'g')
                plt.hlines(15,-20,20,linewidth = .5,colors = 'g')
                plt.vlines(-20,-15,15,linewidth = .5,colors = 'g')
                plt.hlines(-15,-20,20,linewidth = .5,colors = 'g')
                plt.title(nb.EYE[i]+" (°)")
                plt.grid(True,alpha = 0.5)
                plt.scatter(NeurobitDxDev_H.T[i],NeurobitDxDev_V.T[i]+diff_V,
                            s = cir_size,c = 'k',)
                plt.plot(NeurobitDxDev_H.T[i][:-1],NeurobitDxDev_V.T[i][:-1]+diff_V,
                            linewidth = .5,c = 'r',)
                plt.xlim([MIN,MAX])
                plt.ylim([MIN,MAX])
                ax.invert_xaxis()
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
    
            plt.tight_layout()
            plt.savefig(os.path.join(self.saveImage_path,"DrawEyeMesh_"+nb.ACT_LABEL[ss]+".png"), dpi=300)
            plt.close(fig)
            plt.clf()
        