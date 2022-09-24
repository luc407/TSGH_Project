# -*- coding: utf-8 -*-
"""
Created on Wed May 25 23:37:32 2022

@author: luc40
"""

import glob
import os
import math
import cv2
import numpy as np
import pandas as pd 
import qrcode
import sqlite3
import shutil
import time
import re
import sys
from scipy import stats
from tqdm import tqdm
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from PIL import Image
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip, concatenate_videoclips
from function_eye_capture import capture_eye_iris, capture_eye_pupil, get_eye_position
from datetime import datetime
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

# google cloud function
from google.cloud import storage

EYE         = ['OD','OS']
ACT         = 'ACT'
CUT         = 'CUT'
GAZE_9      = '9_Gaze'

DX          = np.array(["control","horizontal strabismus","vertical strabismus",
               "nystagmus","orbital frx","nerve palsy","TAO"])

ACT_TIME    = ["O_t","CL_t", "CR_t",  "UCR_t"]
ACT_COLOR   = ['aqua','plum','gold','lime']
ACT_STR     = ["Open", "Cover Left", "Cover Right", "Uncover"]
ACT_LABEL   = ['O','CL','CR','UCR']

CUT_TIME    = ["O_t", "CL_t", "UCL_t", "CR_t",  "UCR_t"]
CUT_STR     = ["Open", "Cover Left", "Uncover Left", "Cover Right", "Uncover Right"]
CUT_LABEL   = ['O','CL','UCL','CR','UCR']

GAZE_9_TIME     = ['F','U','D','R','L','RU','LU','RD','LD']
GAZE_9_COLOR    = ['deepskyblue',  'wheat',    'mediumpurple', 
                   'slateblue',    'plum',     'lime',         
                   'aqua',         'gold',     'orange']
GAZE_9_STR      = ["Down",       "Front",    "Left",     
                   "Left Down",  "Left Up",  "Right",  
                   "Right Down", "Right Up", "Up"]
GAZE_9_EYEFIG = [5,2,8,4,6,1,3,7,9]

# color map
line_color_palatte = {'greens':["#A5F5B3", "#51F46D",   "#00F62B", "#008D19", "#004D0D"], # pale / mid / base / dark / black              
                      'oranges':["#FFD6AC", "#FFAC54", "#FF8300", "#B95F00", "#653400"],             
                      'reds':["#FFB2AC", "#FF6154", "#FF1300", "#B90D00", "#650700"],                 
                      'blues':["#A4DCEF", "#54C8EE", "#03B5F0", "#015773", "#012F3F"]}

def GetDxXls():
    dx_xls = pd.read_excel(os.path.join(os.getcwd(),'Neurobit database_20210523.xlsx'), dtype=object)
    header = dx_xls.columns.values        
    profile = dx_xls.to_numpy(dtype=str)
    return header[2:], profile

global OD_WTW, OS_WTW, CAL_VAL_OD, CAL_VAL_OS, EYE_ORING, Release_ver
OD_WTW = 0; 
OS_WTW = 0;
CAL_VAL_OD = 5/33;
CAL_VAL_OS = 5/33;
Release_ver = "Release2.01"

def enclosed_area(xy):
    """ 
        Calculates polygon area.
        x = xy[:,0], y = xy[:,1]
    """
    l = len(xy)
    s = 0.0
    # Python arrys are zer0-based
    for i in range(l):
        j = (i+1)%l  # keep index in [0,l)
        s += (xy[j,0] - xy[i,0])*(xy[j,1] + xy[i,1])
    return -0.5*s
    
def trans_PD(AL,dx,CAL_VAL):
    theta = []
    dx = np.array(dx, dtype=float)
    for i in range(0,dx.size):
        try:
            math.asin((2*abs(dx[i])/AL)*(5/33))
        except:
            dx[i] = np.nan
        if dx[i]<0:
            dx[i] = -dx[i]
            theta = np.append(theta, - 100*math.tan(math.asin((2*dx[i]/AL)*CAL_VAL)))
        else:
            theta = np.append(theta, 100*math.tan(math.asin((2*dx[i]/AL)*CAL_VAL)))
    return np.round(theta,1)

def trans_AG(AL,dx,CAL_VAL):
    theta = []
    dx = np.array(dx, dtype=float)
    for i in range(0,dx.size):
        try:
            Th = math.degrees(math.asin((2*abs(dx[i])/AL)*CAL_VAL))
        except:
            try:
                Th = 90+math.degrees(
                    math.asin(
                        (2*(abs(dx[i])-(AL*330)/(2*50))/AL)*CAL_VAL
                        )
                    )
            except:
                Th = np.nan
        if dx[i]<0:
            dx[i] = -dx[i]
            theta = np.append(theta, - Th)
        else:
            theta = np.append(theta, Th)
    return np.round(theta,1)

def GetVideo(csv_path):
    fall = csv_path.replace(".csv",".avi")
    if not os.path.isfile(fall):
        fall = csv_path.replace(".csv",".mp4") 
    return cv2.VideoCapture(fall)

def DrawEyePosition(frame, eyes, OD_p, OS_p):
    for (ex,ey,ew,eh) in eyes:    
        cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    if len(np.argwhere(np.isnan(OD_p)))<3:
        cv2.rectangle(frame,
                      (int(OD_p[0]),int(OD_p[1])),
                      (int(OD_p[0])+1,int(OD_p[1])+1),
                      (0,255,0),2)
        cv2.circle(frame,(int(OD_p[0]),int(OD_p[1])),
                   int(OD_p[2]),
                   (255,255,255),2)        
    if len(np.argwhere(np.isnan(OS_p)))<3:
        cv2.rectangle(frame,
                      (int(OS_p[0]),int(OS_p[1])),
                      (int(OS_p[0])+1,int(OS_p[1])+1),
                      (0,255,0),2)
        cv2.circle(frame,(int(OS_p[0]),int(OS_p[1])),
                   int(OS_p[2]),
                   (255,255,255),2)
class ACT_Save(object):
    _ACT_dx = {'ID':[],
          'Examine Date':[],
          'H_Dx':[],
          'H_Dev':[],
          'H_type':[],
          'V_Dx':[],
          'V_Dev':[],
          }
    header, profile = GetDxXls()
    _Dx_true = dict()
    for i in header:
        _Dx_true[i]=[]
        
class CUT_Save(object):
    _CUT_dx = {'ID':[],
          'Examine Date':[],
          'H_Dx':[],
          'H_Dev':[],
          'H_type':[],
          'V_Dx':[],
          'V_Dev':[],
          }
    header, profile = GetDxXls()
    _Dx_true = dict()
    for i in header:
        _Dx_true[i]=[]

class Gaze9_Save(object):
    _Gaze9_dx = {'ID':[],
          'Examine Date':[]}
    for i in range(0,len(GAZE_9_TIME)):
        _Gaze9_dx[GAZE_9_TIME[i]+'_OD_H_Dev'] = []
        _Gaze9_dx[GAZE_9_TIME[i]+'_OS_H_Dev'] = []
        _Gaze9_dx[GAZE_9_TIME[i]+'_OD_V_Dev'] = []
        _Gaze9_dx[GAZE_9_TIME[i]+'_OS_V_Dev'] = []
    _Gaze9_dx['OD_Area'] = []
    _Gaze9_dx['OS_Area'] = []
    header, profile = GetDxXls()
    _Dx_true = dict()
    for i in header:
        _Dx_true[i]=[]
class Neurobit():
    def __init__(self):
        self.version = '2.8'
        self.version_csv = '2.8'
        self.task = str("Subject")
        self.session = []
        self.major_path = os.getcwd()
        self.save_path = os.getcwd()+"\\RESULT\\Version"+self.version
        self.save_csv_path = os.getcwd()+"\\RESULT\\Version"+self.version_csv
        self.saveVideo_path = []
        self.DB_path = []
        self.CmdTime = []
        self.showVideo = True
        self.AL_OD = 25.15
        self.AL_OS = 25.15
        self.FolderName = []
    def GetFolderPath(self,main_path):        
        return glob.glob(main_path+"\*")
    def GetSubjectFiles(self,folder):
        return glob.glob(folder+"\*.csv")
    def GetDxSql(self):        
        ID  = self.FolderName.split("_")[-1]
        Date = self.FolderName.split("_")[-2]
        Date = datetime(int(Date[:4]), int(Date[4:6]), int(Date[6:])).strftime('%Y/%m/%d')
        con = sqlite3.connect(os.path.join(self.DB_path,"NeurobitNS01-1.db"))
        cur = con.cursor()
        cur.execute("SELECT * FROM Patient WHERE [ID]='" + ID + "'")
# =============================================================================
#         cur.execute('select * from sqlite_master').fetchall()
#         cur.execute('SELECT * FROM'+ID)
# =============================================================================
        profile = np.array(cur.fetchall())[-1]
        cur.execute("SELECT * FROM Visit WHERE [Patient_ID]='" + ID + "'" + 
                    "AND [Datetime]='"+Date+"'"+
                    "AND [Procedure]='OcularMotility'"
                    "AND [Procedure_ID]='0'")
        visit_ID = np.array(cur.fetchall())[0]
        cur.execute("SELECT * FROM ExamSheet WHERE [Visit_ID]='" + visit_ID[0] + "'")
        exam_sheet = np.array(cur.fetchall())[0]
        return ID, profile, exam_sheet, visit_ID
    def GetProfile(self, csv_path):
        global CAL_VAL_OD, CAL_VAL_OS, Release_ver
        cmd_csv = pd.read_csv(csv_path, dtype=object)
        ID, profile, exam_sheet, visit_ID = self.GetDxSql()   
        
        self.Profile_ind = str(cmd_csv.PatientID[0])         
        self.Task   = cmd_csv.Mode[0]        
        self.ID     = cmd_csv.PatientID[0]              
        self.Device = cmd_csv.Device[0]
        self.Doctor = visit_ID[3]
        self.Date   = visit_ID[1].replace("/","")[:8]
        
        tmp = int(np.where(cmd_csv.PatientID == "Eye")[0]+1)
        if self.task == 'ACT':
            self.VoiceCommand = np.array(cmd_csv.PatientID[tmp:], dtype=float)
        elif self.task == '9_Gaze':
            self.VoiceCommand = np.array([cmd_csv.ExaminerID[tmp:],cmd_csv.Device[tmp:],cmd_csv.PatientID[tmp:]], dtype=float)
        elif self.task == 'CUT':
            self.VoiceCommand = np.array(cmd_csv.PatientID[tmp:], dtype=float)
        else:
            pass
        
        if Release_ver == "Release2.00": i =-1
        else: i = 0

        self.Name   = str(profile[4]+profile[2])
        self.Gender = str(profile[6])
        self.DoB    = str(profile[7])
        self.Age    = str(int(self.Date[:4])-int(self.DoB.replace("/","")[:4]))  
        self.Height = str(profile[8])
        
        self.Dx         = str(DX[np.where(exam_sheet[3:10+i]=='True')[0]]) + ", " + exam_sheet[-5] + ", " + visit_ID[-1]
        self.VA_OD      = str(exam_sheet[11+i])
        self.BCVA_OD    = str(exam_sheet[12+i])
        self.Ref_OD     = str(exam_sheet[13+i])
        self.pupil_OD   = str(exam_sheet[14+i])
        try: 
            self.WTW_OD     = float(exam_sheet[15+i]) 
            CAL_VAL_OD      = self.WTW_OD/OD_WTW   
            print("CAL_VAL_OD:%d OD_WTW: %d", CAL_VAL_OD, OD_WTW)
        except: self.WTW_OD  = np.nan
        try: self.AL_OD      = float(exam_sheet[16+i])
        except: self.AL_OD  = np.nan
        
        self.VA_OS      = str(exam_sheet[17+i])
        self.BCVA_OS    = str(exam_sheet[18+i])
        self.Ref_OS     = str(exam_sheet[19+i])
        self.pupil_OS   = str(exam_sheet[20+i])          
        try: 
            self.WTW_OS     = float(exam_sheet[21+i])
            CAL_VAL_OS      = self.WTW_OS/OS_WTW
            print("CAL_VAL_OS:%d OS_WTW: %d", CAL_VAL_OS, OS_WTW)
        except: self.WTW_OS  = np.nan
        try: self.AL_OS      = float(exam_sheet[22+i])
        except: self.AL_OS  = np.nan
        
        self.PD         = str(exam_sheet[23+i])            
        self.Hertal_OD  = str(exam_sheet[24+i])
        self.Hertal_OS  = str(exam_sheet[25+i])
        self.Hertal_Len = str(exam_sheet[26+i])
        self.Stereo     = str(exam_sheet[27+i])
    def SaveDxTrue(self,SaveClass):
        SaveClass._Dx_true['ID'].append(self.ID)
        SaveClass._Dx_true['Name'].append(self.Name)
        SaveClass._Dx_true['Gender'].append(self.Gender)
        SaveClass._Dx_true['Age'].append(self.Age)
        SaveClass._Dx_true['Exam Date'].append(self.Date)
        SaveClass._Dx_true['Diagnosis'].append(self.Dx)
        SaveClass._Dx_true['Height (cm)'].append(self.Height)
        
        SaveClass._Dx_true['VA OD'].append(self.VA_OD)
        SaveClass._Dx_true['VA OS'].append(self.VA_OS)
        SaveClass._Dx_true['BCVA OD'].append(self.BCVA_OD)
        SaveClass._Dx_true['BCVA OS'].append(self.BCVA_OS)
        SaveClass._Dx_true['Ref OD'].append(self.Ref_OD)
        SaveClass._Dx_true['Ref OS'].append(self.Ref_OS)
        
        SaveClass._Dx_true['Steropsis'].append(self.Stereo)
        SaveClass._Dx_true['WTW OD (mm)'].append(self.WTW_OD)
        SaveClass._Dx_true['WTW OS (mm)'].append(self.WTW_OS)
        SaveClass._Dx_true['AL, OD (mm)'].append(self.AL_OD)
        SaveClass._Dx_true['AL, OS (mm)'].append(self.AL_OS)
        
        SaveClass._Dx_true['PD (mm)'].append(self.PD)
        SaveClass._Dx_true['Hertel OD (mm)'].append(self.Hertal_OD)
        SaveClass._Dx_true['Hertel OS (mm)'].append(self.Hertal_OS)
        SaveClass._Dx_true['Hertel length (mm)'].append(self.Hertal_Len)
        
        if "ortho" in self.Dx:SaveClass._Dx_true['Ortho (yes/no)'].append('yes')
        else:SaveClass._Dx_true['Ortho (yes/no)'].append('no')
        
        ind_str = 'xt'
        if ind_str.casefold() in self.Dx.split(",")[1].casefold():
            val = re.findall('\d+.\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'.*\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\s\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall('\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall(ind_str,self.Dx.split(",")[1].lstrip().casefold())
            if not val: sys.exit('No rules for: '+self.Dx.split(",")[1])
            SaveClass._Dx_true['XT (PD)'].append(val[0].replace(ind_str.casefold(),"").replace("^",""))
        else:SaveClass._Dx_true['XT (PD)'].append(0)
        
        ind_str = "et" 
        if ind_str.casefold() in self.Dx.split(",")[1].casefold():
            val = re.findall('\d+.\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'.*\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\s\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall('\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall(ind_str,self.Dx.split(",")[1].lstrip().casefold())
            if not val: sys.exit('No rules for: '+self.Dx.split(",")[1])
            SaveClass._Dx_true['ET (PD)'].append(val[0].replace(ind_str.casefold(),"").replace("^",""))
        else:SaveClass._Dx_true['ET (PD)'].append(0)
        
        ind_str = "lht" 
        if ind_str.casefold() in self.Dx.split(",")[1].casefold():
            val = re.findall('\d+.\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'.*\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\s\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall('\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall(ind_str,self.Dx.split(",")[1].lstrip().casefold())
            if not val: sys.exit('No rules for: '+self.Dx.split(",")[1])
            SaveClass._Dx_true['Hypertropia (PD)'].append(val[0].replace(ind_str.casefold(),"").replace("^",""))
        else:SaveClass._Dx_true['Hypertropia (PD)'].append(0)
        
        ind_str = "lhot" 
        if ind_str.casefold() in self.Dx.split(",")[1].casefold():
            val = re.findall('\d+.\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall('\d+.'+ind_str,self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'.*\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\d+.',self.Dx.split(",")[1].casefold())
            if not val: val = re.findall(ind_str+'\s\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall('\d',self.Dx.split(",")[1].lstrip().casefold())
            if not val: val = re.findall(ind_str,self.Dx.split(",")[1].lstrip().casefold())
            if not val: sys.exit('No rules for: '+self.Dx.split(",")[1])
            SaveClass._Dx_true['Hypotropia (PD)'].append(val[0].replace(ind_str.casefold(),"").replace("^",""))
        else:SaveClass._Dx_true['Hypotropia (PD)'].append(0)
        
        SaveClass._Dx_true['備註'].append(self.Dx)
                            
    def GetEyePosition(self):
        cap = GetVideo(self.csv_path)
        ret, frame = cap.read()
        height = frame.shape[0]
        width = frame.shape[1]

        eyes_origin = [[int(width/4-200),int(height/2)-100,375,200],
                       [int(width/2+75),int(height/2)-100,375,200]]
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        out = cv2.VideoWriter(os.path.join(self.saveVideo_path,self.FileName+'.mp4'),
                          fourcc, 25, (width,height))
        eyes, OD_pre, OS_pre = get_eye_position(GetVideo(self.csv_path),eyes_origin)
        OD = []; OS = []; thr_eyes = [] 
        frame_cnt = 0; OD_cal_cnt = 0; OS_cal_cnt = 0
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap = GetVideo(self.csv_path)
        while(cap.isOpened()):
            cap.set(1,frame_cnt)
            ret, frame = cap.read()
            if ret == True:
                frame_cnt+=1                             
                OD_p,OS_p,thr = capture_eye_pupil(frame,eyes)
                if (not np.isnan(OD_p).any() and 
                    eyes_origin[0][0]<OD_p[0]<eyes_origin[0][0]+eyes_origin[0][2] and 
                    eyes_origin[0][1]<OD_p[1]<eyes_origin[0][1]+eyes_origin[0][3]):
                    OD.append([int(OD_p[0]),int(OD_p[1]), int(OD_p[2])])
                else:
                    OD.append([np.nan,np.nan,np.nan])
                    #print("An OD exception occurred")
                if (not np.isnan(OS_p).any() and 
                    eyes_origin[1][0]<OS_p[0]<eyes_origin[1][0]+eyes_origin[1][2] and 
                    eyes_origin[1][1]<OS_p[1]<eyes_origin[1][1]+eyes_origin[1][3]):
                    OS.append([int(OS_p[0]), int(OS_p[1]), int(OS_p[2])])
                else:
                    OS.append([np.nan,np.nan,np.nan])
                    #print("An OS exception occurred")                
                DrawEyePosition(frame, eyes, OD[-1], OS[-1])
                self.DrawTextVideo(frame, frame_cnt)
                
                for (ex,ey,ew,eh) in eyes_origin:    
                    cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                
                if self.showVideo:
                    cv2.imshow('frame',frame) 
                    cv2.waitKey(1)  
                
                out.write(frame)
                dOD = np.sum(np.abs(np.array(OD[-1])-OD_pre))
                dOS = np.sum(np.abs(np.array(OS[-1])-OS_pre))
                if np.logical_or(dOD>60, np.isnan(dOD)) and OD_cal_cnt <= 0:
                    OD_cal_cnt = 60
                    eyes, OD_pre, OS_pre = get_eye_position(cap,eyes_origin)    
                elif np.logical_or(dOS>60, np.isnan(dOS)) and OD_cal_cnt <= 0 and OS_cal_cnt <= 0:
                    OS_cal_cnt = 60
                    eyes, OD_pre, OS_pre = get_eye_position(cap,eyes_origin) 
                OD_cal_cnt-=1; OS_cal_cnt-=1
                
            else:
                break    
            time.sleep(0.001)
            pbar.update(1)
        if self.showVideo:            
            cv2.destroyAllWindows()
        out.release()
        self.OD = np.array(OD).transpose()
        self.OS = np.array(OS).transpose()
        EyePositionCsv = pd.DataFrame({"OD_x":self.OD[0,:],
                                       "OD_y":self.OD[1,:],
                                       "OD_p":self.OD[2,:],
                                       "OS_x":self.OS[0,:],
                                       "OS_y":self.OS[1,:],
                                       "OS_p":self.OS[2,:]})     
        EyePositionCsv.to_excel(os.path.join(self.saveReport_path,self.task+"_EyePositionCsv.xlsx")) 
    def MergeFile(self):
        if len(self.session)>1:
            csv_1 = pd.read_csv(self.session[0], dtype=object)
            videoList = []
            videoList.append(VideoFileClip(self.session[0].replace(".csv",".mp4")))
            for i in range(1,len(self.session)):                
                csv_2 = pd.read_csv(self.session[i], dtype=object)
                tmp = int(np.where(csv_2.PatientID == "Eye")[0]+1)
                csv_1 = csv_1.append(csv_2[tmp:], ignore_index=True)
                video = VideoFileClip(self.session[i].replace(".csv",".mp4"))
                videoList.append(video)
                              
            final_video = concatenate_videoclips(videoList)
            final_video.write_videofile(os.path.join(self.saveMerge_path,self.FolderName + "_" + self.task + ".mp4"))  
            csv_1.to_csv(os.path.join(self.saveMerge_path,self.FolderName + "_" + self.task + ".csv"))        
            self.csv_path = os.path.join(self.saveMerge_path,self.FolderName + "_" + self.task + ".csv")
            self.FileName = self.csv_path.split('\\')[-1].replace(".csv","")
        else:
            self.csv_path = self.session[0]
    def GetCommand(self):    
        self.GetProfile(self.csv_path)
        self.GetTimeFromCmd()
        self.IsVoiceCommand = True
    def GetACT_Save(self):
        try:
            self.Dx_true = ACT_Save._Dx_true
        except:
            print("No Dx_ture")
        try:
            self.ACT_image_QT = ACT_Save._ACT_image_QT
        except:
            print("No _ACT_image_QT")
        try:
            self.ACT_dx = ACT_Save._ACT_dx     
        except:
            print("No _ACT_dx")
    def GetGaze9_Save(self):
        try:
            self.Gaze9_Dx_true = Gaze9_Save._Dx_true
        except:
            print("No Gaze9_ture")
        try:
            self.Gaze9_dx = Gaze9_Save._Gaze9_dx     
        except:
            print("No _Gaze9_dx")
    
    def GetCUT_Save(self):
        try:
            self.CUT_Dx_true = CUT_Save._Dx_true
        except:
            print("No CUT_ture")
        try:
            self.CUT_dx = CUT_Save._CUT_dx     
        except:
            print("No _CUT_dx")
    def Save2Cloud(self):
        gauth = GoogleAuth()       
        drive = GoogleDrive(gauth) 
        upload_file = self.FileName+".mp4"
       	gfile = drive.CreateFile({'parents': [{'id': '1Sp9f9izaf5580iVP3Sk-jTuy3a84-u0m'}]})
        # delete exist file
        file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1Sp9f9izaf5580iVP3Sk-jTuy3a84-u0m')}).GetList()
        for file in file_list:
            if file['title'] == self.FileName+".mp4":
                drive.CreateFile({'id': file['id']}).Trash()
       	# Read file and set it as the content of this instance.
        os.chdir(self.saveVideo_path)
       	gfile.SetContentFile(upload_file)
        os.chdir(self.major_path)
       	gfile.Upload() # Upload the file.    
        NotUpdated = True
        while NotUpdated:
            file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1Sp9f9izaf5580iVP3Sk-jTuy3a84-u0m')}).GetList()
            for file in file_list:
                if file['title'] == self.FileName+".mp4":
                    NotUpdated = False  
    def DrawQRCode(self):
        os.chdir(self.major_path)
        gauth = GoogleAuth()       
        drive = GoogleDrive(gauth) 
       	# Read file and set it as the content of this instance.
        file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format('1Sp9f9izaf5580iVP3Sk-jTuy3a84-u0m')}).GetList()
        for file in file_list:
            if file['title'] == self.FileName+".mp4":
                self.website =file['alternateLink']
        img = qrcode.make(self.website)
        img.save(os.path.join(self.saveImage_path,"QR_code.png"))
    def Preprocessing(self):
        plt.subplot(2,2,1)        
        plt.plot(self.OD[0,:])
        plt.plot(self.OD[1,:])
        plt.subplot(2,2,2)
        plt.plot(self.OS[0,:])
        plt.plot(self.OS[1,:])
        plt.show()
        win = 15      
        low = np.nanpercentile(self.OD,30,axis=1)
        rm = np.logical_or(abs(self.OD[0,:]-low[0]) > 80, abs(self.OD[1,:]-low[1]) > 80)
        self.OD[:,rm] = np.full([3,len(self.OD[0,rm])],np.nan)
        
        low = np.nanpercentile(self.OS,30,axis=1)
        rm = np.logical_or(abs(self.OS[0,:]-low[0]) > 80, abs(self.OS[1,:]-low[1]) > 80)
        self.OS[:,rm] = np.full([3,len(self.OS[0,rm])],np.nan)
        
        OD_temp = self.OD; OS_temp = self.OS;
        for i in range(win,(len(self.OD[0,:])-win)):
            if not np.isnan(self.OD[0,i]): OD_temp[:,i] = np.nanmedian(self.OD[:,i-win:i+win],axis = 1)
            if not np.isnan(self.OS[0,i]): OS_temp[:,i] = np.nanmedian(self.OS[:,i-win:i+win],axis = 1)
        self.OD = OD_temp
        self.OS = OS_temp
    
class ACT_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "ACT"
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
        self.SaveDxTrue(ACT_Save)
# =============================================================================
#          
#         
#         self.GetEyePosition()  
#         #self.SeperateSession()              
#         self.FeatureExtraction()  
#         self.GetDiagnosis()  
#         self.Save2Cloud()
#         
#         self.DrawEyeFig()
#         self.DrawEyeTrack()  
#         self.DrawQRCode()
# =============================================================================
    def NoVoiceCommandFeatureExtraction(self):
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        OD_ACT = []; OS_ACT = [];       # all position in each ACT_TIME  
        tmp = np.concatenate([np.array(self.CmdTime[ACT_TIME[1]]),
                        np.array(self.CmdTime[ACT_TIME[2]])])

        up_OD = np.nanpercentile(OD[:,tmp].astype(int),80, axis =1)
        low_OD = np.nanpercentile(OD[:,tmp].astype(int),20, axis =1)
        
        up_OS = np.nanpercentile(OS[:,tmp].astype(int),80, axis =1)
        low_OS = np.nanpercentile(OS[:,tmp].astype(int),20, axis =1)
        
        for i in range(0,len(ACT_TIME)):
            temp = self.CmdTime[ACT_TIME[i]]
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)                
            if temp.any():                
                OD_ACT.append(stats.mode(OD[:,temp].astype(int),axis = 1)[0].reshape(-1))
                OS_ACT.append(stats.mode(OS[:,temp].astype(int),axis = 1)[0].reshape(-1))
            else:
                OD_ACT.append([np.nan, np.nan, np.nan])
                OS_ACT.append([np.nan, np.nan, np.nan])
        
        CL = OD_ACT[1][:2]; CR = OD_ACT[2][:2]
        if (np.nansum(abs(CL-CR))>6):
            if sum(abs(CL-up_OD[:2])) < sum(abs(CR-up_OD[:2])):
                OD_ACT[1] = up_OD; OD_ACT[2] = low_OD
            else:
                OD_ACT[2] = up_OD; OD_ACT[1] = low_OD
            
        CL = OS_ACT[1][:2]; CR = OS_ACT[2][:2]    
        if (np.nansum(abs(CL-CR))>6):
            if sum(abs(CL-up_OS[:2])) < sum(abs(CR-up_OS[:2])):
                OS_ACT[1] = up_OS; OS_ACT[2] = low_OS
            else:
                OS_ACT[2] = up_OS; OS_ACT[1] = low_OS
        
        # ET、XT angle
        self.OD_ACT = np.array(np.round(OD_ACT,2))
        self.OS_ACT = np.array(np.round(OS_ACT,2))
        
        # Fixation_eye - Covered_eye
        OD_fix = self.OD_ACT[1]-self.OD_ACT[2]    # CL-CR
        OS_fix = self.OS_ACT[2]-self.OS_ACT[1]    # CR-CL
        
        try:
            OD_fix = np.append(trans_PD(self.AL_OD,OD_fix[0:2],CAL_VAL_OD), OD_fix[2])
            OS_fix = np.append(trans_PD(self.AL_OS,OS_fix[0:2],CAL_VAL_OS), OS_fix[2])
        except:
            print("No profile")
            
        self.OD_fix = OD_fix        # one position in each ACT_TIME
        self.OS_fix = OS_fix        
    def VoiceCommandFeatureExtraction(self):
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        OD_ACT = []; OS_ACT = [];       # all position in each ACT_TIME  
        OD_ACT_CL = []; OS_ACT_CL = [];
        OD_ACT_CR = []; OS_ACT_CR = [];
        O_trg_ind = np.where(np.diff(self.CmdTime['O_t']) > 5)[0]
        UCR_trg_ind = np.where(np.diff(self.CmdTime['UCR_t']) > 5)[0]
        CL_trg_ind = np.where(np.diff(self.CmdTime['CL_t']) > 5)[0]
        CR_trg_ind = np.where(np.diff(self.CmdTime['CR_t']) > 5)[0]
        i = 0; rd = 0; rd_ucr = 0; duration = 5*24
        while(i < len(OD[0,:])):
            if i in self.CmdTime['O_t']:
                tmp = self.CmdTime['O_t']
                if rd > 0 and rd < len(O_trg_ind): 
                    i = tmp[O_trg_ind[rd]]+1;        rd += 1
                    print(i)
                    continue
                elif rd > 0:
                    i = self.CmdTime['O_t'][-1]+1;   rd += 1
                    print(i)
                    continue
                OD_ACT.append(stats.mode(OD[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                OS_ACT.append(stats.mode(OS[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                
                i = self.CmdTime['CL_t'][0]            
                rd += 1
                print("O_t",i)
            elif i in self.CmdTime['CL_t'] or i in self.CmdTime['CR_t']:   
                
                if i in self.CmdTime['CL_t']: tmp = self.CmdTime['CL_t'];print("CL_t")                  
                else: tmp = self.CmdTime['CR_t'];print("CR_t")               
                
                start_ind = np.where(tmp == i)[0][0]
                if i in self.CmdTime['CL_t']: end_ind = CL_trg_ind[np.where(CL_trg_ind>start_ind)[0]]
                else: end_ind = CR_trg_ind[np.where(CR_trg_ind>start_ind)[0]]
                
                if end_ind.any(): end_ind = end_ind[0]
                else: end_ind = len(tmp)-1
                
                latency = np.nanmean(OD[:2,tmp[start_ind:start_ind+10]],axis = 1).reshape(2,-1)
                slope = np.nansum(abs(OD[:2,tmp[start_ind:end_ind]]-latency),axis=0)
                trg_ind = np.where(slope>2.5)[0]
                if trg_ind.any(): 
                    trg_ind = tmp[start_ind] + trg_ind[0]
                    if i in self.CmdTime['CL_t']: 
                        OD_ACT_CL.append(stats.mode(OD[:,trg_ind:trg_ind+duration].astype(int),axis = 1)[0].reshape(-1))
                    else: 
                        OD_ACT_CR.append(stats.mode(OD[:,trg_ind:trg_ind+duration].astype(int),axis = 1)[0].reshape(-1))
                else:
                    if i in self.CmdTime['CL_t']: 
                        OD_ACT_CL.append(stats.mode(OD[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                    else: 
                        OD_ACT_CR.append(stats.mode(OD[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                
                latency = np.nanmean(OS[:2,tmp[start_ind:start_ind+10]],axis = 1).reshape(2,-1)
                slope = np.nansum(abs(OS[:2,tmp[start_ind:end_ind]]-latency),axis=0)
                trg_ind = np.where(slope>2.5)[0]
                if trg_ind.any(): 
                    trg_ind = tmp[start_ind] + trg_ind[0]
                    if i in self.CmdTime['CL_t']: 
                        OS_ACT_CL.append(stats.mode(OS[:,trg_ind:trg_ind+duration].astype(int),axis = 1)[0].reshape(-1))
                    else: 
                        OS_ACT_CR.append(stats.mode(OS[:,trg_ind:trg_ind+duration].astype(int),axis = 1)[0].reshape(-1))
                else:
                    if i in self.CmdTime['CL_t']: 
                        OS_ACT_CL.append(stats.mode(OS[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                    else: 
                        OS_ACT_CR.append(stats.mode(OS[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                
                i = tmp[end_ind]+1
                print(i)
            elif i in self.CmdTime['UCR_t']:
                tmp = self.CmdTime['UCR_t']
                if rd_ucr > 0 and rd_ucr < len(UCR_trg_ind): 
                    i = tmp[UCR_trg_ind[rd_ucr]]+1;        rd_ucr += 1
                    print(i)
                    continue
                elif rd_ucr > 0:
                    i = self.CmdTime['UCR_t'][-1]+1;     rd_ucr += 1
                    print(i)
                    continue
                OD_ACT.append(stats.mode(OD[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                OS_ACT.append(stats.mode(OS[:,tmp].astype(int),axis = 1)[0].reshape(-1))
                
                if UCR_trg_ind.any(): i = tmp[UCR_trg_ind[rd_ucr]]+1
                else: i = self.CmdTime['UCR_t'][-1]+1                
                rd_ucr += 1
                print("UCR",i)
            else:
                i+=1
        OD_ACT_CL_n = stats.mode(OD_ACT_CL,axis = 0)[0].reshape(-1)
        OS_ACT_CL_n = stats.mode(OS_ACT_CL,axis = 0)[0].reshape(-1)
        OD_ACT_CR_n = stats.mode(OD_ACT_CR,axis = 0)[0].reshape(-1)
        OS_ACT_CR_n = stats.mode(OS_ACT_CR,axis = 0)[0].reshape(-1)
        # ET、XT angle
        OD_ACT = np.insert(OD_ACT,1,OD_ACT_CL_n, axis=0)
        OD_ACT = np.insert(OD_ACT,2,OD_ACT_CR_n, axis=0)
        OS_ACT = np.insert(OS_ACT,1,OS_ACT_CL_n, axis=0)
        OS_ACT = np.insert(OS_ACT,2,OS_ACT_CR_n, axis=0)
        self.OD_ACT = np.array(np.round(OD_ACT,2))
        self.OS_ACT = np.array(np.round(OS_ACT,2))   
        # Fixation_eye - Covered_eye
        OD_fix_tmp = []; OS_fix_tmp = []
        for i in range(0,len(OD_ACT_CR)):
            try: OD_fix_tmp.append(OD_ACT_CL[2*(i+1)-2]-OD_ACT_CR[i])
            except: pass
            try: OD_fix_tmp.append(OD_ACT_CL[2*(i+1)-1]-OD_ACT_CR[i])
            except: pass
            try: OS_fix_tmp.append(OS_ACT_CR[i]-OS_ACT_CL[2*(i+1)-2])
            except: pass
            try: OS_fix_tmp.append(OS_ACT_CR[i]-OS_ACT_CL[2*(i+1)-1])
            except: pass
            
        OD_fix = stats.mode(np.array(OD_fix_tmp).astype(int),axis = 0)[0].reshape(-1)    # CL-CR
        OS_fix = stats.mode(np.array(OS_fix_tmp).astype(int),axis = 0)[0].reshape(-1)     # CR-CL
        
        try:
            OD_fix = np.append(trans_PD(self.AL_OD,OD_fix[0:2],CAL_VAL_OD), OD_fix[2])
            OS_fix = np.append(trans_PD(self.AL_OS,OS_fix[0:2],CAL_VAL_OS), OS_fix[2])
        except:
            print("No profile")
            
        self.OD_fix = OD_fix        # one position in each ACT_TIME
        self.OS_fix = OS_fix 
        
    def FeatureExtraction(self):  
        delete = np.where(self.CmdTime['O_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['O_t'] = np.delete(self.CmdTime['O_t'], delete)
        
        delete = np.where(self.CmdTime['CL_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['CL_t'] = np.delete(self.CmdTime['CL_t'], delete)
        
        delete = np.where(self.CmdTime['CR_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['CR_t'] = np.delete(self.CmdTime['CR_t'], delete)
        
        delete = np.where(self.CmdTime['UCR_t']>len(self.OD[0])-1)[0]
        if delete.any():
            self.CmdTime['UCR_t'] = np.delete(self.CmdTime['UCR_t'], delete)
            
        if not self.IsVoiceCommand:
            self.NoVoiceCommandFeatureExtraction()
        else:           
            self.VoiceCommandFeatureExtraction()   
        plt.subplot(2,2,1)
        plt.title(self.ID)
        plt.plot(self.OD[0,:]-np.nanmin(self.OD[0,:]))
        plt.plot(self.OD[1,:]-np.nanmin(self.OD[1,:]))
        plt.hlines(self.OD_ACT[1][0]-np.nanmin(self.OD[0,:]),0,len(self.OD[1,:]), color ='r')# CL
        plt.hlines(self.OD_ACT[1][1]-np.nanmin(self.OD[1,:]),0,len(self.OD[1,:]), color ='g')# CL
        plt.hlines(self.OD_ACT[2][0]-np.nanmin(self.OD[0,:]),0,len(self.OD[1,:]), color ='r')# CR
        plt.hlines(self.OD_ACT[2][1]-np.nanmin(self.OD[1,:]),0,len(self.OD[1,:]), color ='g')# CR
        plt.subplot(2,2,2)
        plt.plot(self.OS[0,:]-np.nanmin(self.OS[0,:]))
        plt.plot(self.OS[1,:]-np.nanmin(self.OS[1,:]))
        plt.hlines(self.OS_ACT[1][0]-np.nanmin(self.OS[0,:]),0,len(self.OS[1,:]), color ='r')# CL
        plt.hlines(self.OS_ACT[1][1]-np.nanmin(self.OS[1,:]),0,len(self.OS[1,:]), color ='g')# CL
        plt.hlines(self.OS_ACT[2][0]-np.nanmin(self.OS[0,:]),0,len(self.OS[1,:]), color ='r')# CR
        plt.hlines(self.OS_ACT[2][1]-np.nanmin(self.OS[1,:]),0,len(self.OS[1,:]), color ='g')# CR
        plt.show()
    def GetDiagnosis(self):
        self.SaveDxTrue(ACT_Save)
        OD_fix = self.OD_fix; OS_fix = self.OS_fix
        thr =1.5
        self.NeurobitDx_H = None
        self.NeurobitDx_V = None
        self.NeurobitDxTp_X = None
        if np.all(np.abs([OD_fix,OS_fix])<=thr):
            self.Ortho = True
            self.NeurobitDx_H = 'Ortho'
            self.NeurobitDx_V = 'Ortho'
            self.NeurobitDxTp_H = 'None'
            self.NeurobitDxDev_H = 0
            self.NeurobitDxDev_V = 0
        else:
            self.Ortho = False
        
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
        ACT_Save._ACT_dx['ID'].append(self.ID)
        ACT_Save._ACT_dx['Examine Date'].append(self.Date)
        ACT_Save._ACT_dx['H_Dx'].append(self.NeurobitDx_H)
        ACT_Save._ACT_dx['H_Dev'].append(self.NeurobitDxDev_H)
        ACT_Save._ACT_dx['H_type'].append(self.NeurobitDxTp_X)
        ACT_Save._ACT_dx['V_Dx'].append(self.NeurobitDx_V)
        ACT_Save._ACT_dx['V_Dev'].append(self.NeurobitDxDev_V)
    def GetTimeFromCmd(self):
        cmd = self.VoiceCommand
        O_t = np.where(cmd==0)[0]
        CL_t = np.where(np.logical_or(cmd==1,cmd==3))[0]
        CR_t = np.where(cmd==2)[0] 
        UCL_t = np.where(cmd==4)[0]
        self.CmdTime = {"CL_t": np.array(CL_t),
                        "CR_t": np.array(CR_t),
                        "O_t":  np.array(O_t),
                        "UCR_t":np.array(UCL_t)}
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(ACT_TIME)):
            temp = np.array(self.CmdTime[ACT_TIME[i]])
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
                self.CmdTime[ACT_TIME[i]] = list_temp
            else:
                self.CmdTime[ACT_TIME[i]] = temp
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/25
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(EYE)):
            if EYE[i] == 'OD':
                x_diff = self.OD_ACT[0,0]-OD[0,:]
                y_diff = self.OD_ACT[0,1]-OD[1,:]
                x_PD = trans_PD(self.AL_OD,x_diff,CAL_VAL_OD)
                y_PD = trans_PD(self.AL_OD,y_diff,CAL_VAL_OD)
            else:
                x_diff = self.OS_ACT[0,0]-OS[0,:]
                y_diff = self.OS_ACT[0,1]-OS[1,:]
                x_PD = trans_PD(self.AL_OS,x_diff,CAL_VAL_OS)
                y_PD = trans_PD(self.AL_OS,y_diff,CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position (PD)")
            plt.title("Alternated Cover Test "+ EYE[i])
            
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
            try:t = np.concatenate(np.array(self.CmdTime[ACT_TIME[i]]))
            except:t = self.CmdTime[ACT_TIME[i]]
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
                #ind = np.where(Diff == np.nanmin(Diff))[0]
                #ind_pu = np.where(pupil[ind] == np.nanmax(pupil[ind]))[0]
                ACT.append(t[np.where(Diff == np.nanmin(Diff))[0][0]])
            except:
                ACT.append(ACT[-1])
                #print("Not Detect "+ ACT_TIME[i]) 
        pic_cont = 1
        empt=0
        #fig = plt.figure(figsize=(11.7,8.3))
        fig = plt.gcf()
        fig.set_size_inches(3,4.4, forward=True)
        fig.set_dpi(300)
        for pic in ACT:
            cap = GetVideo(self.csv_path)
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
            exec('ax'+str(pic_cont)+'=plt.subplot(4, 1, pic_cont)')
            exec('ax'+str(pic_cont)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
            exec('ax'+str(pic_cont)+'.axes.xaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.axes.yaxis.set_ticks([])')
            exec('ax'+str(pic_cont)+ '.set_ylim(int(3*height/4),int(height/4))')
# =============================================================================
#             exec('ax'+str(pic_cont)+ '.set_ylim(int(height),int(0))')
# =============================================================================
            exec('ax'+str(pic_cont)+ '.set_ylabel(ACT_LABEL[pic_cont-1])')
            plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]
        for i in range(0,len(ACT_TIME)):
            if frame_cnt in self.CmdTime[ACT_TIME[i]]:
                text = ACT_STR[i]
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
    
            
class Gaze9_Task(Neurobit):
    def __init__(self, csv_path):
        Neurobit.__init__(self)
        self.task = "9_Gaze"
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
        self.GetCommand() 
        self.SaveDxTrue(Gaze9_Save)
# =============================================================================
#         Finished_ACT = False
#         for arg in args:
#             if arg: 
#                 Finished_ACT = True
#                 ACT_Task = arg
#           
#         
#         self.GetEyePosition()
#         #self.SeperateSession()                
#         if Finished_ACT: self.FeatureExtraction(ACT_Task) 
#         else: self.FeatureExtraction() 
#         self.Save2Cloud()
#                
#         self.DrawEyeFig()
#         self.DrawEyeMesh()
#         self.DrawEyeTrack()  
#         self.DrawQRCode() 
# =============================================================================
    def GetTimeFromCmd(self):    
        #if int(self.Date) >= 20210601:
        #    stupid_bug = np.where(self.VoiceCommand[0,:] == self.VoiceCommand[0,0])[0]
        #    self.VoiceCommand[:,stupid_bug] = 0
    
        x = np.round(self.VoiceCommand[0,:],2); y = np.round(self.VoiceCommand[1,:],2); self.CmdTime = dict()
        cover = self.VoiceCommand[2,:]
        
        x_q1 = np.unique(x)[0]
        x_q2 = np.unique(x)[1]
        x_q3 = np.unique(x)[-1]

        y_q1 = np.unique(y)[0]
        y_q2 = np.unique(y)[1]
        y_q3 = np.unique(y)[-1]
        
        D = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), y == y_q1))[0]#[60:]
        F = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), np.logical_and(y < y_q3, y > y_q1)))[0]#[60:]
        L = np.where(np.logical_and(np.logical_and(x == x_q1, y < y_q3), y > y_q1))[0]#[60:]
        LD = np.where(np.logical_and(x == x_q1, y == y_q1))[0]#[60:]
        LU = np.where(np.logical_and(x == x_q1, y == y_q3))[0]#[60:]
        R = np.where(np.logical_and(np.logical_and(x == x_q3 ,  y < y_q3), y > y_q1))[0]#[60:]
        RD = np.where(np.logical_and(x == x_q3, y == y_q1))[0]#[60:]
        RU = np.where(np.logical_and(x == x_q3, y == y_q3))[0]#[60:]
        U = np.where(np.logical_and(np.logical_and(x < x_q3, x > x_q1), y == y_q3))[0]#[60:]
        
        for i in range(0,len(GAZE_9_TIME)):
            exec('cov_non = np.where(cover['+GAZE_9_TIME[i]+'] == 0)[0]')
            exec('self.CmdTime[GAZE_9_TIME[i]] = '+GAZE_9_TIME[i]+'[cov_non]')
    def FeatureExtraction(self,*args):
        GAZE_9_TIME     = ['D','F','L','LD','LU','R','RD','RU','U']
        OD = self.OD; OS = self.OS; Finished_ACT = False
        Gaze_9_OD = []; Gaze_9_OS = [];       # all position in each ACT_TIME
        NeurobitDxDev_H = list();NeurobitDxDev_V = list()
        for i in range(0,len(GAZE_9_TIME)):
            temp = self.CmdTime[GAZE_9_TIME[i]]
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)
            if type(self.CmdTime[GAZE_9_TIME[i]]) == list:
                OD_ACT_tmp = []; OS_ACT_tmp = [];
                for temp in self.CmdTime[GAZE_9_TIME[i]]:
                    if i == 0: 
                        OD_ACT_tmp.append([np.nanpercentile(OD[0,temp],50),
                                           np.nanpercentile(OD[1,temp],80),
                                           np.nanpercentile(OD[2,temp],50)])
                        OS_ACT_tmp.append([np.nanpercentile(OS[0,temp],50),
                                           np.nanpercentile(OS[1,temp],80),
                                           np.nanpercentile(OS[2,temp],50)])
                    elif i == 1: 
                        OD_ACT_tmp.append(np.nanpercentile(OD[0,temp],50,axis = 1))
                        OS_ACT_tmp.append(np.nanpercentile(OS[0,temp],50,axis = 1))
                    elif i == 2:
                        OD_ACT_tmp.append([np.nanpercentile(OD[0,temp],80),
                                           np.nanpercentile(OD[1,temp],50),
                                           np.nanpercentile(OD[2,temp],50)])
                        OS_ACT_tmp.append([np.nanpercentile(OS[0,temp],80),
                                           np.nanpercentile(OS[1,temp],50),
                                           np.nanpercentile(OS[2,temp],50)])
                    elif i == 3:
                        OD_ACT_tmp.append(np.nanpercentile(OD[0,temp],80,axis = 1))
                        OS_ACT_tmp.append(np.nanpercentile(OS[0,temp],80,axis = 1))
                    elif i == 4:
                        OD_ACT_tmp.append([np.nanpercentile(OD[0,temp],80),
                                           np.nanpercentile(OD[1,temp],20),
                                           np.nanpercentile(OD[2,temp],50)])
                        OS_ACT_tmp.append([np.nanpercentile(OS[0,temp],80),
                                           np.nanpercentile(OS[1,temp],20),
                                           np.nanpercentile(OS[2,temp],50)])
                    elif i == 5:
                        OD_ACT_tmp.append([np.nanpercentile(OD[0,temp],20),
                                           np.nanpercentile(OD[1,temp],50),
                                           np.nanpercentile(OD[2,temp],50)])
                        OS_ACT_tmp.append([np.nanpercentile(OS[0,temp],20),
                                           np.nanpercentile(OS[1,temp],50),
                                           np.nanpercentile(OS[2,temp],50)])
                    elif i == 6:
                        OD_ACT_tmp.append([np.nanpercentile(OD[0,temp],20),
                                           np.nanpercentile(OD[1,temp],80),
                                           np.nanpercentile(OD[2,temp],50)])
                        OS_ACT_tmp.append([np.nanpercentile(OS[0,temp],20),
                                           np.nanpercentile(OS[1,temp],80),
                                           np.nanpercentile(OS[2,temp],50)])
                    elif i == 7:
                        OD_ACT_tmp.append(np.nanpercentile(OD[0,temp],20,axis = 1))
                        OS_ACT_tmp.append(np.nanpercentile(OS[0,temp],20,axis = 1))
                    elif i == 8:
                        OD_ACT_tmp.append([np.nanpercentile(OD[0,temp],50),
                                           np.nanpercentile(OD[1,temp],20),
                                           np.nanpercentile(OD[2,temp],50)])
                        OS_ACT_tmp.append([np.nanpercentile(OS[0,temp],50),
                                           np.nanpercentile(OS[1,temp],20),
                                           np.nanpercentile(OS[2,temp],50)])
                OD_ACT_tmp = np.array(OD_ACT_tmp)
                OS_ACT_tmp = np.array(OS_ACT_tmp)
                Gaze_9_OD.append(np.round(stats.mode(OD_ACT_tmp.astype(int),axis = 1)[0][0].reshape(-1),2))
                Gaze_9_OS.append(np.round(stats.mode(OS_ACT_tmp.astype(int),axis = 1)[0][0].reshape(-1),2))
            elif temp.any():                
                if i == 0: 
                    Gaze_9_OD.append([np.nanpercentile(OD[0,temp],50),
                                       np.nanpercentile(OD[1,temp],80),
                                       np.nanpercentile(OD[2,temp],50)])
                    Gaze_9_OS.append([np.nanpercentile(OS[0,temp],50),
                                       np.nanpercentile(OS[1,temp],80),
                                       np.nanpercentile(OS[2,temp],50)])
                elif i == 1: 
                    Gaze_9_OD.append(np.nanpercentile(OD[:,temp],50,axis = 1))
                    Gaze_9_OS.append(np.nanpercentile(OS[:,temp],50,axis = 1))
                elif i == 2:
                    Gaze_9_OD.append([np.nanpercentile(OD[0,temp],80),
                                       np.nanpercentile(OD[1,temp],50),
                                       np.nanpercentile(OD[2,temp],50)])
                    Gaze_9_OS.append([np.nanpercentile(OS[0,temp],80),
                                       np.nanpercentile(OS[1,temp],50),
                                       np.nanpercentile(OS[2,temp],50)])
                elif i == 3:
                    Gaze_9_OD.append(np.nanpercentile(OD[:,temp],80,axis = 1))
                    Gaze_9_OS.append(np.nanpercentile(OS[:,temp],80,axis = 1))
                elif i == 4:
                    Gaze_9_OD.append([np.nanpercentile(OD[0,temp],80),
                                       np.nanpercentile(OD[1,temp],20),
                                       np.nanpercentile(OD[2,temp],50)])
                    Gaze_9_OS.append([np.nanpercentile(OS[0,temp],80),
                                       np.nanpercentile(OS[1,temp],20),
                                       np.nanpercentile(OS[2,temp],50)])
                elif i == 5:
                    Gaze_9_OD.append([np.nanpercentile(OD[0,temp],20),
                                       np.nanpercentile(OD[1,temp],50),
                                       np.nanpercentile(OD[2,temp],50)])
                    Gaze_9_OS.append([np.nanpercentile(OS[0,temp],20),
                                       np.nanpercentile(OS[1,temp],50),
                                       np.nanpercentile(OS[2,temp],50)])
                elif i == 6:
                    Gaze_9_OD.append([np.nanpercentile(OD[0,temp],20),
                                       np.nanpercentile(OD[1,temp],80),
                                       np.nanpercentile(OD[2,temp],50)])
                    Gaze_9_OS.append([np.nanpercentile(OS[0,temp],20),
                                       np.nanpercentile(OS[1,temp],80),
                                       np.nanpercentile(OS[2,temp],50)])
                elif i == 7:
                    Gaze_9_OD.append(np.nanpercentile(OD[:,temp],20,axis = 1))
                    Gaze_9_OS.append(np.nanpercentile(OS[:,temp],20,axis = 1))
                elif i == 8:
                    Gaze_9_OD.append([np.nanpercentile(OD[0,temp],50),
                                       np.nanpercentile(OD[1,temp],20),
                                       np.nanpercentile(OD[2,temp],50)])
                    Gaze_9_OS.append([np.nanpercentile(OS[0,temp],50),
                                       np.nanpercentile(OS[1,temp],20),
                                       np.nanpercentile(OS[2,temp],50)])
            else:
                Gaze_9_OD.append([np.nan, np.nan, np.nan])
                Gaze_9_OS.append([np.nan, np.nan, np.nan])
                        
        self.Gaze_9_OD = np.array(np.round(Gaze_9_OD,2))
        self.Gaze_9_OS = np.array(np.round(Gaze_9_OS,2))
        
        for arg in args:
            if arg:
                Finished_ACT = True
                ACT_Task = arg
            
        for i in range(0,len(GAZE_9_TIME)):
            if Finished_ACT:
                diff_OD_x = ACT_Task.OD_ACT[1][0]-self.Gaze_9_OD[i][0]
                diff_OD_y = ACT_Task.OD_ACT[1][1]-self.Gaze_9_OD[i][1]
                diff_OS_x = ACT_Task.OS_ACT[2][0]-self.Gaze_9_OS[i][0]
                diff_OS_y = ACT_Task.OS_ACT[2][1]-self.Gaze_9_OS[i][1]
            else:
                diff_OD_x = self.Gaze_9_OD[1][0]-self.Gaze_9_OD[i][0]
                diff_OD_y = self.Gaze_9_OD[1][1]-self.Gaze_9_OD[i][1]
                diff_OS_x = self.Gaze_9_OS[1][0]-self.Gaze_9_OS[i][0]
                diff_OS_y = self.Gaze_9_OS[1][1]-self.Gaze_9_OS[i][1]
            PD_OD = trans_AG(self.AL_OD,np.array([diff_OD_x, diff_OD_y]),CAL_VAL_OD)
            PD_OS = trans_AG(self.AL_OS,np.array([diff_OS_x, diff_OS_y]),CAL_VAL_OS)
            NeurobitDxDev_H.append([PD_OD[0], PD_OS[0]])
            NeurobitDxDev_V.append([PD_OD[1], PD_OS[1]])
            Gaze9_Save._Gaze9_dx[GAZE_9_TIME[i]+'_OD_H_Dev'].append(PD_OD[0])
            Gaze9_Save._Gaze9_dx[GAZE_9_TIME[i]+'_OS_H_Dev'].append(PD_OS[0])
            Gaze9_Save._Gaze9_dx[GAZE_9_TIME[i]+'_OD_V_Dev'].append(PD_OD[1])
            Gaze9_Save._Gaze9_dx[GAZE_9_TIME[i]+'_OS_V_Dev'].append(PD_OS[1])
        self.NeurobitDxDev_H = np.round(np.array(NeurobitDxDev_H),2)
        self.NeurobitDxDev_V = np.round(np.array(NeurobitDxDev_V),2)
        
        GAZE_9_TIME     = ['D','F','L','LD','LU','R','RD','RU','U']
        a = self.NeurobitDxDev_H.transpose()[0][[5,7,8,4,2,3,0,6]]
        b = self.NeurobitDxDev_V.transpose()[0][[5,7,8,4,2,3,0,6]]
        xy = np.array([a,b]).transpose()
        OD_Area = np.round(enclosed_area(xy),2)
        
        a = self.NeurobitDxDev_H.transpose()[1][[5,7,8,4,2,3,0,6]]
        b = self.NeurobitDxDev_V.transpose()[1][[5,7,8,4,2,3,0,6]]
        xy = np.array([a,b]).transpose()
        OS_Area = np.round(enclosed_area(xy),2)
        
        Gaze9_Save._Gaze9_dx['ID'].append(self.ID)
        Gaze9_Save._Gaze9_dx['Examine Date'].append(self.Date)
        Gaze9_Save._Gaze9_dx['OD_Area'].append(OD_Area)
        Gaze9_Save._Gaze9_dx['OS_Area'].append(OS_Area)       
    def SeperateSession(self):
        OD = self.OD; OS = self.OS
        for i in range(0,len(GAZE_9_TIME)):
            temp = self.CmdTime[GAZE_9_TIME[i]]
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
                self.CmdTime[GAZE_9_TIME[i]] = list_temp
            else:
                self.CmdTime[GAZE_9_TIME[i]] = temp
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]      
        for i in range(0,len(GAZE_9_TIME)):
            if frame_cnt in self.CmdTime[GAZE_9_TIME[i]]:
                text = GAZE_9_STR[i]
                textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 2, 2)[0]
                textX = int((width - textsize[0]) / 2)
                cv2.putText(frame,text, (textX, 100), 
                            cv2.FONT_HERSHEY_TRIPLEX, 
                            2, (255, 255, 255),
                            2, cv2.LINE_AA)        
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/25
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(EYE)):
            if EYE[i] == 'OD':
                x_diff = self.Gaze_9_OD[0][0]-OD[0,:]
                y_diff = self.Gaze_9_OD[0][1]-OD[1,:]
                x_PD = trans_AG(self.AL_OD,x_diff,CAL_VAL_OD)
                y_PD = trans_AG(self.AL_OD,y_diff,CAL_VAL_OD)
            else:
                x_diff = self.Gaze_9_OS[0][0]-OS[0,:]
                y_diff = self.Gaze_9_OS[0][1]-OS[1,:]
                x_PD = trans_AG(self.AL_OS,x_diff,CAL_VAL_OS)
                y_PD = trans_AG(self.AL_OS,y_diff,CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position ("+chr(176)+")")
            plt.title("9 Gaze Test "+ EYE[i])
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
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeTrack.png"), dpi=300) 
        plt.close()
    def DrawEyeFig(self):
        Gaze_9 = []; OD = self.OD; OS = self.OS
        for i in range(0,len(self.Gaze_9_OD)):
            try:t = np.concatenate(np.array(self.CmdTime[GAZE_9_TIME[i]]))
            except:t = self.CmdTime[GAZE_9_TIME[i]]
            if not np.isnan(self.Gaze_9_OD[i,0]) and not np.isnan(self.Gaze_9_OS[i,0]):
                OD_diff = abs(OD[0,t]-self.Gaze_9_OD[i,0])+abs(OD[1,t]-self.Gaze_9_OD[i,1])
                OS_diff = abs(OS[0,t]-self.Gaze_9_OS[i,0])+abs(OS[1,t]-self.Gaze_9_OS[i,1])
                Diff = np.sum(np.array([OD_diff, OS_diff]),axis = 0)
            elif np.isnan(self.Gaze_9_OD[i,0]):
                Diff = abs(OS[0,t]-self.Gaze_9_OS[i,0])+abs(OS[1,t]-self.Gaze_9_OS[i,1])
            else:
                Diff = abs(OD[0,t]-self.Gaze_9_OD[i,0])+abs(OD[1,t]-self.Gaze_9_OD[i,1])                
            if not np.isnan(Diff).all():
                Gaze_9.append(t[np.where(Diff == np.nanmin(Diff))[0][0]])
            else:
                Gaze_9.append(np.nan)
        pic_cont = 0
        empt=0
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)
        for pic in Gaze_9:
            if not np.isnan(pic):
                cap = GetVideo(self.csv_path)
                cap.set(1,pic)
                ret, im = cap.read()
                height = im.shape[0]
                width = im.shape[1]
                try:
                    cv2.rectangle(im,
                                  (int(self.Gaze_9_OD[pic_cont][0]),int(self.Gaze_9_OD[pic_cont][1])),
                                  (int(self.Gaze_9_OD[pic_cont][0])+1,int(self.Gaze_9_OD[pic_cont][1])+1),
                                  (0,255,0),2)
                    cv2.circle(im,(int(self.Gaze_9_OD[pic_cont][0]),int(self.Gaze_9_OD[pic_cont][1])),
                               int(self.Gaze_9_OD[pic_cont][2]),
                               (255,255,255),2) 
                except:
                    pass#print("OD Absent!")
                try:
                    cv2.rectangle(im,
                                  (int(self.Gaze_9_OS[pic_cont][0]),int(self.Gaze_9_OS[pic_cont][1])),
                                  (int(self.Gaze_9_OS[pic_cont][0])+1,int(self.Gaze_9_OS[pic_cont][1])+1),
                                  (0,255,0),2)
                    cv2.circle(im,(int(self.Gaze_9_OS[pic_cont][0]),int(self.Gaze_9_OS[pic_cont][1])),
                               int(self.Gaze_9_OS[pic_cont][2]),
                               (255,255,255),2)
                except:
                    pass#print("OS Absent!")
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                _,thresh_1 = cv2.threshold(gray,110,255,cv2.THRESH_TRUNC)
                
                exec('ax'+str(pic_cont+1)+'=plt.subplot(3, 3, GAZE_9_EYEFIG[pic_cont])')
                exec('ax'+str(pic_cont+1)+ '.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), "gray")')
                exec('ax'+str(pic_cont+1)+'.axes.xaxis.set_ticks([])')
                exec('ax'+str(pic_cont+1)+ '.axes.yaxis.set_ticks([])')
                exec('ax'+str(pic_cont+1)+ '.set_ylim(int(3*height/4),int(height/4))')
# =============================================================================
#                 exec('ax'+str(pic_cont+1)+ '.set_ylim(int(height),int(0))')
# =============================================================================
                exec('ax'+str(pic_cont+1)+ '.set_ylabel(GAZE_9_TIME[pic_cont])')
                plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
        plt.close()
    def DrawEyeMesh(self):
        border = [7,2,8,4,6,1,5,3,7]
        MIN = -50; MAX = 50
        fig = plt.gcf()
        fig.set_size_inches(6/1.2,3/1.2, forward=True)
        fig.set_dpi(300)
        cir_size = 25
        try:
            diff_V = np.round(math.degrees(math.atan(abs(170-self.Height)/220)),2)
            if self.Height>170:
                diff_V = -diff_V
        except:
            diff_V = 0
        for i in range(0,len(EYE)):
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
            plt.title(EYE[i]+" (°)")
            plt.grid(True,alpha = 0.5)
            plt.scatter(-self.NeurobitDxDev_H[:,i],self.NeurobitDxDev_V[:,i]+diff_V,
                        s = cir_size,c = 'k',)
            plt.plot(-self.NeurobitDxDev_H[border,i],self.NeurobitDxDev_V[border,i]+diff_V,
                        linewidth = .5,c = 'r',)
            plt.xlim([MIN,MAX])
            plt.ylim([MIN,MAX])
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeMesh.png"), dpi=300)
        plt.close()
        
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
        self.SaveDxTrue(CUT_Save)
# =============================================================================
#          
#         
#         self.GetEyePosition()  
#         self.SeperateSession()              
#         self.FeatureExtraction()  
#         self.GetDiagnosis()  
#         self.Save2Cloud()
#         
#         self.DrawEyeFig()
#         self.DrawEyeTrack()  
#         self.DrawQRCode()
# =============================================================================
    def FeatureExtraction(self):
        OD = self.OD.astype('float'); OS = self.OS.astype('float')
        OD_ACT = []; OS_ACT = [];       # all position in each CUT_TIME
        for i in range(0,len(CUT_TIME)):
            temp = self.CmdTime[CUT_TIME[i]]
            delete = np.where(temp>len(OD[0])-1)[0]
            if delete.any():
                temp = np.delete(temp, delete)
            if type(self.CmdTime[CUT_TIME[i]]) == list:
                OD_ACT_tmp = []; OS_ACT_tmp = [];
                for temp in self.CmdTime[CUT_TIME[i]]:
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
            OD_fix = np.append(trans_PD(self.AL_OD,OD_fix[0:2],CAL_VAL_OD), OD_fix[2])
            OS_fix = np.append(trans_PD(self.AL_OS,OS_fix[0:2],CAL_VAL_OS), OS_fix[2])
            
            OS_phoria = np.append(trans_PD(self.AL_OS,OS_phoria[0:2],CAL_VAL_OS), OS_phoria[2])
            OD_phoria = np.append(trans_PD(self.AL_OD,OD_phoria[0:2],CAL_VAL_OD), OD_phoria[2])
        except:
            print("No profile")
        self.OD_fix = OD_fix        # one position in each CUT_TIME
        self.OS_fix = OS_fix
        self.OD_phoria = OD_phoria        # one position in each CUT_TIME
        self.OS_phoria = OS_phoria
    def GetDiagnosis(self):
        self.SaveDxTrue(CUT_Save)
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
        
        CUT_Save._CUT_dx['ID'].append(self.ID)
        CUT_Save._CUT_dx['Examine Date'].append(self.Date)
        CUT_Save._CUT_dx['H_Dx'].append(self.NeurobitDx_H)
        CUT_Save._CUT_dx['H_Dev'].append(self.NeurobitDxDev_H)
        CUT_Save._CUT_dx['H_type'].append(self.NeurobitDxTp_X)
        CUT_Save._CUT_dx['V_Dx'].append(self.NeurobitDx_V)
        CUT_Save._CUT_dx['V_Dev'].append(self.NeurobitDxDev_V)
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
        for i in range(0,len(CUT_TIME)):
            temp = np.array(self.CmdTime[CUT_TIME[i]])
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
                self.CmdTime[CUT_TIME[i]] = list_temp
            else:
                self.CmdTime[CUT_TIME[i]] = temp
    def DrawEyeTrack(self):
        OD = self.OD; OS = self.OS
        time = np.array(range(0,len(OD[0])))/25
        fig = plt.gcf()
        fig.set_size_inches(7.2,2.5, forward=True)
        fig.set_dpi(300)              
        for i in range(0,len(EYE)):
            if EYE[i] == 'OD':
                x_diff = self.OD_ACT[0,0]-OD[0,:]
                y_diff = self.OD_ACT[0,1]-OD[1,:]
                x_PD = trans_PD(self.AL_OD,x_diff,CAL_VAL_OD)
                y_PD = trans_PD(self.AL_OD,y_diff,CAL_VAL_OD)
            else:
                x_diff = self.OS_ACT[0,0]-OS[0,:]
                y_diff = self.OS_ACT[0,1]-OS[1,:]
                x_PD = trans_PD(self.AL_OS,x_diff,CAL_VAL_OS)
                y_PD = trans_PD(self.AL_OS,y_diff,CAL_VAL_OS)
            plt.subplot(1,2,i+1)
            plt.plot(time,x_PD, linewidth=1, color = 'b',label = 'X axis')
            plt.plot(time,y_PD, linewidth=1, color = 'r',label = 'Y axis')
            
            plt.xlabel("Time (s)")
            plt.ylabel("Eye Position (PD)")
            plt.title("Cover Uncover Test "+ EYE[i])
            
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
            try:t = np.concatenate(np.array(self.CmdTime[CUT_TIME[i]]))
            except:t = self.CmdTime[CUT_TIME[i]]
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
            cap = GetVideo(self.csv_path)
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
            exec('ax'+str(pic_cont)+ '.set_ylabel(CUT_LABEL[pic_cont-1])')
            plt.box(on=None)
            pic_cont+=1
        plt.tight_layout()
        plt.savefig(os.path.join(self.saveImage_path,"DrawEyeFig.png"), dpi=300)
    def DrawTextVideo(self, frame, frame_cnt):
        width = frame.shape[1]
        for i in range(0,len(CUT_TIME)):
            if frame_cnt in self.CmdTime[CUT_TIME[i]]:
                text = CUT_STR[i]
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
        
        