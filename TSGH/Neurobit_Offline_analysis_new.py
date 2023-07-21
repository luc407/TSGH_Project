# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:09:44 2022

@author: luc40
"""

import os
import numpy as np
import pandas as pd 
import subprocess
import Neurobit
import shutil
import time
from tqdm import tqdm
from Gaze9_Task import Gaze9_Task
from Gaze9ACT_Task import Gaze9ACT_Task
from ACT_Task import ACT_Task
from CUT_Task import CUT_Task
from Neurobit import Neurobit as NB
from Neurobit import ACT_Save, CUT_Save, Gaze9_Save
from datetime import datetime
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1,\
    ACTReport, CreatePDF, Gaze9Report, CUTReport,Gaze9ACTReport
from calibration import CalibSystem



main_path = os.getcwd()
Fail = []
Subject = NB()
fialList = []
            
if __name__== '__main__':    
    Neurobit.Release_ver = "Release2.01"
    main_path = "E:\\Result\\"+ Neurobit.Release_ver +"\\Result"
    Subject.DB_path = "E:\\Result\\"+ Neurobit.Release_ver
    folderList = Subject.GetFolderPath(main_path)
    pbar = tqdm(total=len(folderList))
    for folder1 in folderList:        
        sub_folderList = Subject.GetFolderPath(folder1)        
        for folder2 in sub_folderList:
            try: del ACT_task
            except: pass
            try: del Gaze9_task
            except: pass 
            try: del CUT_task
            except: pass
            csv_files = Subject.GetSubjectFiles(folder2)      
            IsACT_Task = False; IsGaze9_Task = False; IsCUT_Task = False
            for csv_path in csv_files:
                Subject.FolderName = csv_path.split("\\")[-2]
                Subject.GetProfile(csv_path)
                
                if "9 Gaze Motility Test (9Gaze)" in Subject.Task:
                    try: Gaze9_task.session.append(csv_path)
                    except:
                        Gaze9_task = Gaze9ACT_Task(csv_path)   
                        Gaze9_task.session.append(csv_path)
                elif "Alternate Cover" in Subject.Task:
                    try: ACT_task.session.append(csv_path)
                    except:
                        ACT_task = ACT_Task(csv_path)  
                        ACT_task.session.append(csv_path)
                elif "Cover/Uncover Test (CUT)" in Subject.Task:
                    try: CUT_task.session.append(csv_path)
                    except:
                        CUT_task = CUT_Task(csv_path)  
                        CUT_task.session.append(csv_path)
                else:
                    pass
        
            
# =============================================================================
#             try: ACT_task; IsACT_Task = True
#             except: print("No ACT_Task!!!")
#             if IsACT_Task:
#                 ACT_task.MergeFile()                
#                 eye_file = os.path.join(ACT_task.save_path, ACT_task.FolderName+"\\"+ACT_task.task+"_EyePositionCsv.xlsx")
#                 df = pd.read_excel(eye_file)
#                 ACT_task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
#                 ACT_task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
#                 ACT_task.Preprocessing()
#                 ACT_task.GetCommand()
#                 ACT_task.SaveDxTrue(ACT_Save)
# # =============================================================================
# #                 ACT_task.SeperateSession()
# # =============================================================================
#                 ACT_task.FeatureExtraction()
#                 ACT_task.GetDiagnosis()
#                 #ACT_task.Save2Cloud()
#                 
#                 ACT_task.DrawEyeFig()
#                 ACT_task.DrawEyeTrack()  
#                 #ACT_task.DrawQRCode()
# =============================================================================
            
# =============================================================================
#             try: CUT_task; IsCUT_Task = True
#             except: print("No CUT_Task!!!")
#             if IsCUT_Task:
#                 CUT_task.MergeFile()
#                 eye_file = os.path.join(CUT_task.save_path, CUT_task.FolderName+"\\"+ CUT_task.task+"_EyePositionCsv.xlsx")
#                 df = pd.read_excel(eye_file)
#                 CUT_task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
#                 CUT_task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
#                 CUT_task.Preprocessing()
#                 CUT_task.GetCommand()
#                 CUT_task.SaveDxTrue(CUT_Save)
# # =============================================================================
# #                 CUT_task.SeperateSession()
# # =============================================================================
#                 CUT_task.FeatureExtraction()
#                 CUT_task.GetDiagnosis()
#                 #CUT_task.Save2Cloud()
#                 #CUT_task.GetQuality()
#                 
#                 CUT_task.DrawEyeFig()
#                 CUT_task.DrawEyeTrack()  
#                 #CUT_task.DrawQRCode()
# =============================================================================
            
            try: Gaze9_task; IsGaze9_Task = True
            except: print("No Gaze9_Task!!!")
            if IsGaze9_Task:
                try:
                    Gaze9_task.MergeFile()
                    eye_file = os.path.join(Gaze9_task.save_path,  Gaze9_task.FolderName+"\\"+"9_Gaze"+"_EyePositionCsv.xlsx")
                    df = pd.read_excel(eye_file)
                    Gaze9_task.OD = np.array([df.OD_x, df.OD_y, df.OD_p])
                    Gaze9_task.OS = np.array([df.OS_x, df.OS_y, df.OS_p])
                    Gaze9_task.Preprocessing()
                    Gaze9_task.GetCommand()
# =============================================================================
#                     Gaze9_task.SaveDxTrue(Gaze9_Save)
# =============================================================================
    # =============================================================================
    #                 Gaze9_task.SeperateSession()        
    # =============================================================================
                    Gaze9_task.FeatureExtraction()        
                    #Gaze9_task.Save2Cloud()
            
                    Gaze9_task.DrawEyeFig()
                    Gaze9_task.DrawEyeMesh()
                    Gaze9_task.DrawEyeTrack() 
                    #Gaze9_task.DrawQRCode()
                except:
                    fialList.append(folder2)
                    IsGaze9_Task = False
                
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
                    pdf_path = os.path.join(Gaze9_task.saveReport_path, Gaze9_task.FolderName+"_report.pdf")
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
                        Gaze9ACTReport(Element, Gaze9_task)
                    except:
                        pass 
                    pdf.build(Element)
                    
                    shutil.copy(pdf_path, os.path.join("E:\Peggy_analysis\ALL_REPORT",pdf_path.split("\\")[-1])) 
# =============================================================================
#                     subprocess.Popen(pdf_path, shell=True)
# =============================================================================
                    del pdf
        time.sleep(0.001)
        pbar.update(1)           
    
# =============================================================================
#     Subject.GetACT_Save()
#     ACT_dx_pd = pd.DataFrame(Subject.ACT_dx) 
#     ACT_Dx_true_pd = pd.DataFrame.from_dict(Subject.Dx_true, orient='index').transpose() 
#     ACT_dx_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_ACT_dx.xlsx'))   
#     ACT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_ACT_dx_true.xlsx'))  
# =============================================================================
    
    Subject.GetGaze9_Save()
    Gaze9_dx_pd = pd.DataFrame.from_dict(Subject.Gaze9_dx, orient='index').transpose() 
    Gaze9_Dx_true_pd = pd.DataFrame.from_dict(Subject.Gaze9_Dx_true, orient='index').transpose() 
    Gaze9_dx_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_Gaze9ACT_dx.xlsx'))   
    Gaze9_Dx_true_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_Gaze9ACT_Dx_true.xlsx'))
# =============================================================================
#     
#     Subject.GetCUT_Save()
#     CUT_dx_pd = pd.DataFrame(Subject.CUT_dx) 
#     CUT_Dx_true_pd = pd.DataFrame.from_dict(Subject.Dx_true, orient='index').transpose() 
#     CUT_dx_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_CUT_dx.xlsx'))   
#     CUT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,Neurobit.Release_ver+'_CUT_Dx_true.xlsx'))
# =============================================================================
        
# =============================================================================
#     ACT_image_QT_pd = pd.DataFrame({ key:pd.Series(value) for key, value in Subject.ACT_image_QT.items() }) 
#     ACT_image_QT_pd.to_excel(os.path.join(Subject.save_path,'ACT_image_QT_new.xlsx'))  
# 
# =============================================================================
