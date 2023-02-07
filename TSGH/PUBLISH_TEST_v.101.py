# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 21:52:55 2022

@author: luc40
"""

import glob
import os
import math
import numpy as np
import pandas as pd 
import Neurobit_Lib
import shutil
from matplotlib import pyplot as plt
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report, CUTReport

Fail = []
Subject = Neurobit_Lib.Neurobit()
            
if __name__== '__main__':
    main_path = "E:\Result\ALL"
    folderList = Subject.GetFolderPath(main_path)
    for folder in folderList[50:51]:
        try: del ACT_Task
        except: pass
        try: del Gaze9_Task
        except: pass 
        try: del CUT_Task
        except: pass       
        IsACT_Task = False; IsGaze9_Task = False; IsCUT_Task = False
        csv_files = Subject.GetSubjectFiles(folder)        
        for csv_path in csv_files:
            cal_read_path = os.path.join(Subject.save_csv_path,csv_path.split("\\")[-2]+"//cal_param.txt")
            f = open(cal_read_path)
            text = f.readlines()            
            Neurobit_Lib.OD_WTW = int(text[0].replace("\n",""))
            Neurobit_Lib.OS_WTW = int(text[1].replace("\n",""))
            f.close
            Subject.GetProfile(csv_path)
            if ("Cover/Uncover" in Subject.Task or "Calibration" in Subject.Task) and int(Subject.Date) < 20210601: 
                try:
                    ACT_Task.session.append(csv_path)
                except:
                    ACT_Task = Neurobit_Lib.ACT_Task(csv_path)  
                    ACT_Task.session.append(csv_path)
            elif Subject.Task == "13 : Alternate Cover (ACT)":
                try:
                    ACT_Task.session.append(csv_path)
                except:
                    ACT_Task = Neurobit_Lib.ACT_Task(csv_path)  
                    ACT_Task.session.append(csv_path)
            elif "Cover/Uncover" in Subject.Task and int(Subject.Date) > 20210601:
                try:
                    CUT_Task.session.append(csv_path)
                except:
                    CUT_Task = Neurobit_Lib.CUT_Task(csv_path) 
                    CUT_Task.session.append(csv_path)
            elif "9 Gaze" in Subject.Task:
                try:
                    Gaze9_Task.session.append(csv_path)
                except:
                    Gaze9_Task = Neurobit_Lib.Gaze9_Task(csv_path)   
                    Gaze9_Task.session.append(csv_path)               
        
        try: ACT_Task; IsACT_Task = True
        except: print("No ACT_Task!!!")
        if IsACT_Task:
            ACT_Task.MergeFile()
            eye_file = os.path.join(ACT_Task.save_csv_path, ACT_Task.FolderName+"\\"+ACT_Task.task+"_EyePositionCsv.xlsx")
            df = pd.read_excel(eye_file)
            ACT_Task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
            ACT_Task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
            ACT_Task.Preprocessing()
            ACT_Task.GetCommand()
            ACT_Task.FeatureExtraction()
            ACT_Task.GetDiagnosis()
        
        try: CUT_Task; IsCUT_Task = True
        except: print("No CUT_Task!!!")
        if IsCUT_Task:
            CUT_Task.MergeFile()
            eye_file = os.path.join(CUT_Task.save_csv_path, CUT_Task.FolderName+"\\"+ CUT_Task.task+"_EyePositionCsv.xlsx")
            df = pd.read_excel(eye_file)
            CUT_Task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
            CUT_Task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
            CUT_Task.Preprocessing()
            CUT_Task.GetCommand()
            CUT_Task.FeatureExtraction()
            CUT_Task.GetDiagnosis()
        
        try: Gaze9_Task; IsGaze9_Task = True
        except: print("No Gaze9_Task!!!")
        if IsGaze9_Task:
            Gaze9_Task.MergeFile()
            eye_file = os.path.join(Gaze9_Task.save_csv_path,  Gaze9_Task.FolderName+"\\"+Gaze9_Task.task+"_EyePositionCsv.xlsx")
            df = pd.read_excel(eye_file)
            Gaze9_Task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
            Gaze9_Task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
            Gaze9_Task.Preprocessing()  
            Gaze9_Task.GetCommand()
            try: Gaze9_Task.FeatureExtraction(ACT_Task)
            except: Gaze9_Task.FeatureExtraction()    
            Gaze9_Task.DrawEyeMesh()
    
    Subject.GetACT_Save()
    ACT_dx_pd = pd.DataFrame(Subject.ACT_dx) 
    ACT_Dx_true_pd = pd.DataFrame(Subject.Dx_true) 
    ACT_dx_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx.xlsx'))   
    ACT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx_true.xlsx'))  
    
    Subject.GetGaze9_Save()
    Gaze9_dx_pd = pd.DataFrame(Subject.Gaze9_dx) 
    Gaze9_Dx_true_pd = pd.DataFrame(Subject.Gaze9_Dx_true)
    Gaze9_dx_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_dx.xlsx'))   
    Gaze9_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_Dx_true.xlsx'))
    
    Subject.GetCUT_Save()
    CUT_dx_pd = pd.DataFrame(Subject.CUT_dx) 
    CUT_Dx_true_pd = pd.DataFrame(Subject.CUT_Dx_true)
    CUT_dx_pd.to_excel(os.path.join(Subject.save_path,'CUT_dx.xlsx'))   
    CUT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'CUT_Dx_true.xlsx'))


def poly_plot(xy, titlestr = "", margin = 0.25):
    """ 
        Plots polygon. For arrow see:
        http://matplotlib.org/examples/pylab_examples/arrow_simple_demo.html
        x = xy[:,0], y = xy[:,1]
    """
    xmin = np.min(xy[:,0])
    xmax = np.max(xy[:,0])
    ymin = np.min(xy[:,1])
    ymax = np.max(xy[:,1])
    hl = 0.1
    l = len(xy)
    for i in range(l):
        j = (i+1)%l  # keep index in [0,l)
        dx = xy[j,0] - xy[i,0]
        dy = xy[j,1] - xy[i,1]
        dd = np.sqrt(dx*dx + dy*dy)
        dx = dx*(1 - hl/dd)
        dy = dy*(1 - hl/dd)
        plt.arrow(xy[i,0], xy[i,1], dx, dy, head_width=0.05, head_length=0.1, fc='b', ec='b')
        plt.xlim(xmin-margin, xmax+margin)
        plt.ylim(ymin-margin, ymax+margin)
    plt.title(titlestr)
    
    