# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 21:36:47 2021

@author: luc40
"""

import glob
import os
import math
import numpy as np
import pandas as pd 
import Neurobit_Lib
from matplotlib import pyplot as plt
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report

Fail = []
Subject = Neurobit_Lib.Neurobit()
QT_thr = {"ID":[],
          "OD":[],
          "OS":[]}
            
if __name__== '__main__':
    main_path = "E:\Result\ALL"
    folderList = Subject.GetFolderPath(main_path)
    for folder in folderList:
        try: del ACT_Task
        except: pass
        try: del Gaze9_Task
        except: pass        
        IsACT_Task = False; IsGaze9_Task = False; 
        csv_files = Subject.GetSubjectFiles(folder)        
        for csv_path in csv_files:
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
            eye_file = os.path.join(ACT_Task.saveReport_path, ACT_Task.task+"_EyePositionCsv.xlsx")
            df = pd.read_excel(eye_file)
            ACT_Task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
            ACT_Task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
            QT_thr["ID"].append(Subject.ID)            
            QT_thr["OD"].append(np.nanmean(df.OD_thr))
            QT_thr["OS"].append(np.nanmean(df.OS_thr))
            ACT_Task.GetCommand()
            ACT_Task.SeperateSession()
            ACT_Task.FeatureExtraction()
            ACT_Task.GetDiagnosis()
            #ACT_Task.Save2Cloud()
            #ACT_Task.GetQuality()
            
            ACT_Task.DrawEyeFig()
            ACT_Task.DrawEyeTrack()  
            #ACT_Task.DrawQRCode()
        
        try: Gaze9_Task; IsGaze9_Task = True
        except: print("No Gaze9_Task!!!")
        if IsGaze9_Task:
            Gaze9_Task.MergeFile()
            eye_file = os.path.join(Gaze9_Task.saveReport_path, Gaze9_Task.task+"_EyePositionCsv.xlsx")
            df = pd.read_excel(eye_file)
            Gaze9_Task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
            Gaze9_Task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
            Gaze9_Task.GetCommand()
            #Gaze9_Task.SeperateSession()        
            try: Gaze9_Task.FeatureExtraction(ACT_Task)
            except: Gaze9_Task.FeatureExtraction()        
            #Gaze9_Task.Save2Cloud()
    
            Gaze9_Task.DrawEyeFig()
            Gaze9_Task.DrawEyeMesh()
            Gaze9_Task.DrawEyeTrack() 
            #Gaze9_Task.DrawQRCode()
        
        """Plot Report"""
        
        if IsACT_Task or IsGaze9_Task:
            PDF_Header = main_head("Neurobit")
            try:
                Subject_Table = subject_table(ACT_Task)
                Clinic_Table = clinic_table(ACT_Task)
                pdf = CreatePDF(os.path.join(ACT_Task.saveReport_path, ACT_Task.FolderName+"_report.pdf"))
            except:
                Subject_Table = subject_table(Gaze9_Task)
                Clinic_Table = clinic_table(Gaze9_Task)
                pdf = CreatePDF(os.path.join(Gaze9_Task.saveReport_path, Gaze9_Task.FolderName+"_report.pdf"))
    
            Sub_Header = sub_head("Clinical relevant data")     
            
            Element = []
            Element.append(PDF_Header)
            Element.append(Subject_Table)
            Element.append(Sub_Header)
            Element.append(Clinic_Table)
            try:
                ACTReport(Element, ACT_Task)
            except:
                pass
            Element.append(PageBreak())
            try:
                Gaze9Report(Element, Gaze9_Task)
            except:
                pass 
            pdf.build(Element)
            
            del pdf
           
    
    Subject.GetACT_Save()
    Subject.GetGaze9_Save()
    ACT_image_QT_pd = pd.DataFrame({ key:pd.Series(value) for key, value in Subject.ACT_image_QT.items() }) 
    ACT_dx_pd = pd.DataFrame(Subject.ACT_dx) 
    ACT_Dx_true_pd = pd.DataFrame(Subject.Dx_true) 
    QT_thr_pd = pd.DataFrame(QT_thr)
    
    Gaze9_dx_pd = pd.DataFrame(Subject.Gaze9_dx) 
    Gaze9_Dx_true_pd = pd.DataFrame(Subject.Gaze9_Dx_true)
    
    ACT_image_QT_pd.to_excel(os.path.join(Subject.save_path,'ACT_image_QT.xlsx'))  
    ACT_dx_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx.xlsx'))   
    ACT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx_true.xlsx'))
    QT_thr_pd.to_excel(os.path.join(Subject.save_path,'QT_thr.xlsx'))
    Gaze9_dx_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_dx.xlsx'))   
    Gaze9_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_Dx_true.xlsx'))
        
