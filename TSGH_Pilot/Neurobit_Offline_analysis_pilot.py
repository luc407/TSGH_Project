# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:09:44 2022

@author: luc40
"""

import os
import numpy as np
import pandas as pd 
import subprocess
import shutil
from Gaze9_Task import Gaze9_Task
from ACT_Task import ACT_Task
from CUT_Task import CUT_Task
from Neurobit import Neurobit
from datetime import datetime
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report, CUTReport
from calibration import CalibSystem

Fail = []
Subject = Neurobit()
            
if __name__== '__main__':
    Neurobit.Release_ver = "Release2.02_pilot"
    main_path = "E:\\Result\\"+ Neurobit.Release_ver +"\\Result"
    Subject.DB_path = "E:\\Result\\"+ Neurobit.Release_ver
    folderList = Subject.GetFolderPath(main_path)
    for folder1 in folderList:
        try: del ACT_task
        except: pass
        try: del Gaze9_task
        except: pass 
        try: del CUT_task
        except: pass
        sub_folderList = Subject.GetFolderPath(folder1)
        IsACT_Task = False; IsGaze9_Task = False; IsCUT_Task = False
        for folder2 in sub_folderList:            
            csv_files = Subject.GetSubjectFiles(folder2)        
            for csv_path in csv_files:
                Subject.FolderName = csv_path.split("\\")[-2]
                Subject.GetProfile(csv_path)
                if "Alternate Cover" in Subject.Task:
                    try:
                        ACT_task.session.append(csv_path)
                    except:
                        ACT_task = ACT_Task(csv_path)  
                        ACT_task.session.append(csv_path)
                elif "Cover/Uncover Test (CUT)" in Subject.Task:
                    try:
                        CUT_task.session.append(csv_path)
                    except:
                        CUT_task = CUT_Task(csv_path) 
                        CUT_task.session.append(csv_path)
                elif "9 Gaze Motility Test (9Gaze)" in Subject.Task:
                    try:
                        Gaze9_task.session.append(csv_path)
                    except:
                        Gaze9_task = Gaze9_Task(csv_path)   
                        Gaze9_task.session.append(csv_path)               
            
            try: ACT_task; IsACT_Task = True
            except: print("No ACT_Task!!!")
            if IsACT_Task:
                ACT_task.MergeFile()
                eye_file = os.path.join(ACT_task.save_csv_path, ACT_task.FolderName+"\\"+ACT_task.task+"_EyePositionCsv.xlsx")
                df = pd.read_excel(eye_file)
                ACT_task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
                ACT_task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
                ACT_task.Preprocessing()
                ACT_task.GetCommand()
# =============================================================================
#                 ACT_task.SeperateSession()
# =============================================================================
                ACT_task.FeatureExtraction()
                ACT_task.GetDiagnosis()
                #ACT_task.Save2Cloud()
                
                ACT_task.DrawEyeFig()
                ACT_task.DrawEyeTrack()  
                #ACT_task.DrawQRCode()
            
            try: CUT_task; IsCUT_Task = True
            except: print("No CUT_Task!!!")
            if IsCUT_Task:
                CUT_task.MergeFile()
                eye_file = os.path.join(CUT_task.save_csv_path, CUT_task.FolderName+"\\"+ CUT_task.task+"_EyePositionCsv.xlsx")
                df = pd.read_excel(eye_file)
                CUT_task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
                CUT_task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
                CUT_task.Preprocessing()
                CUT_task.GetCommand()
# =============================================================================
#                 CUT_task.SeperateSession()
# =============================================================================
                CUT_task.FeatureExtraction()
                CUT_task.GetDiagnosis()
                #CUT_task.Save2Cloud()
                #CUT_task.GetQuality()
                
                CUT_task.DrawEyeFig()
                CUT_task.DrawEyeTrack()  
                #CUT_task.DrawQRCode()
            
            try: Gaze9_task; IsGaze9_Task = True
            except: print("No Gaze9_Task!!!")
            if IsGaze9_Task:
                Gaze9_task.MergeFile()
                eye_file = os.path.join(Gaze9_task.save_csv_path,  Gaze9_task.FolderName+"\\"+Gaze9_task.task+"_EyePositionCsv.xlsx")
                df = pd.read_excel(eye_file)
                Gaze9_task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
                Gaze9_task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
                Gaze9_task.Preprocessing()
                Gaze9_task.GetCommand()
# =============================================================================
#                 Gaze9_task.SeperateSession()        
# =============================================================================
                try: Gaze9_task.FeatureExtraction(ACT_task)
                except: Gaze9_task.FeatureExtraction()        
                #Gaze9_task.Save2Cloud()
        
                Gaze9_task.DrawEyeFig()
                Gaze9_task.DrawEyeMesh()
                Gaze9_task.DrawEyeTrack() 
                #Gaze9_task.DrawQRCode()
            
            """Plot Report"""        
            PDF_Header = main_head("Neurobit")
            if IsACT_Task:
                Subject_Table = subject_table(ACT_task)
                Clinic_Table = clinic_table(ACT_task)
                pdf = CreatePDF(os.path.join(ACT_task.saveReport_path, ACT_task.FolderName+"_report.pdf"))
                pdf_path = os.path.join(ACT_task.saveReport_path, ACT_task.FolderName+"_report.pdf")
            elif IsGaze9_Task:
                Subject_Table = subject_table(Gaze9_task)
                Clinic_Table = clinic_table(Gaze9_task)
                pdf = CreatePDF(os.path.join(Gaze9_task.saveReport_path, Gaze9_task.FolderName+"_report.pdf"))
                pdf_path = os.path.join(Gaze9_task.saveReport_path, Gaze9_Task.FolderName+"_report.pdf")
            if IsCUT_Task:
                Subject_Table = subject_table(CUT_task)
                Clinic_Table = clinic_table(CUT_task)
                pdf = CreatePDF(os.path.join(CUT_task.saveReport_path, CUT_task.FolderName+"_report.pdf"))
                pdf_path = os.path.join(CUT_task.saveReport_path, CUT_task.FolderName+"_report.pdf")            
            
            if IsACT_Task or IsGaze9_Task or IsCUT_Task:
                Sub_Header = sub_head("Clinical relevant data")     
                
                Element = []
                Element.append(PDF_Header)
                Element.append(Subject_Table)
                Element.append(Sub_Header)
                Element.append(Clinic_Table)
                try:
                    ACTReport(Element, ACT_task)
                except:
                    pass
                Element.append(PageBreak())
                try:
                    CUTReport(Element, CUT_task)
                except:
                    pass
                Element.append(PageBreak())
                try:
                    Gaze9Report(Element, Gaze9_task)
                except:
                    pass 
                pdf.build(Element)
                
                shutil.copy(pdf_path, os.path.join("E:\Peggy_analysis\ALL_REPORT",pdf_path.split("\\")[-1])) 
# =============================================================================
#                 subprocess.Popen(pdf_path, shell=True)
# =============================================================================
                del pdf
           
    
    Subject.GetACT_Save()
    ACT_dx_pd = pd.DataFrame(Subject.ACT_dx) 
    ACT_dx_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_ACT_dx.xlsx'))   
    
    Subject.GetGaze9_Save()
    Gaze9_dx_pd = pd.DataFrame(Subject.Gaze9_dx) 
    Gaze9_dx_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_Gaze9_dx.xlsx'))   
    
    Subject.GetCUT_Save()
    CUT_dx_pd = pd.DataFrame(Subject.CUT_dx) 
    CUT_dx_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_CUT_dx.xlsx'))   
        
# =============================================================================
#     ACT_image_QT_pd = pd.DataFrame({ key:pd.Series(value) for key, value in Subject.ACT_image_QT.items() }) 
#     ACT_image_QT_pd.to_excel(os.path.join(Subject.save_path,'ACT_image_QT_new.xlsx'))  
# 
# =============================================================================
