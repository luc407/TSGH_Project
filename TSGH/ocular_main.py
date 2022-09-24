# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:19:18 2021

@author: luc40
"""
from cmath import nan
import glob
import os
import math
import sqlite3
import numpy as np
import pandas as pd 
import subprocess
import shutil
import tkinter as tk
import Neurobit_Lib_new
from datetime import datetime
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report, CUTReport
from calibration import CalibSystem



main_path = os.getcwd()
Fail = []
Subject = Neurobit_Lib_new.Neurobit()
            
if __name__== '__main__':    
    Neurobit_Lib_new.Release_ver = "Release2.01"
    main_path = "E:\\Result\\0816\\Release2.01\\Result"
    Subject.DB_path = "E:\\Result\\0816\\Release2.01"
    folderList = Subject.GetFolderPath(main_path)
    for folder1 in folderList:        
        sub_folderList = Subject.GetFolderPath(folder1)
        IsACT_Task = False; IsGaze9_Task = False; IsCUT_Task = False
        for folder2 in sub_folderList:  
            try: del ACT_Task
            except: pass
            try: del Gaze9_Task
            except: pass 
            try: del CUT_Task
            except: pass
            csv_files = Subject.GetSubjectFiles(folder2)        
            for csv_path in csv_files:
                Subject.FolderName = csv_path.split("\\")[-2]
                Subject.GetProfile(csv_path)
                
                if "9 Gaze Motility Test (9Gaze)" in Subject.Task:
                    try: Gaze9_Task.session.append(csv_path)
                    except:
                        Gaze9_Task = Neurobit_Lib_new.Gaze9_Task(csv_path)   
                        Gaze9_Task.session.append(csv_path)
                elif "Alternate Cover" in Subject.Task:
                    try: ACT_Task.session.append(csv_path)
                    except:
                        ACT_Task = Neurobit_Lib_new.ACT_Task(csv_path)  
                        ACT_Task.session.append(csv_path)
                elif "Cover/Uncover Test (CUT)" in Subject.Task:
                    try: CUT_Task.session.append(csv_path)
                    except:
                        CUT_Task = Neurobit_Lib_new.CUT_Task(csv_path)  
                        CUT_Task.session.append(csv_path)
                else:
                    pass
            
            try: ACT_Task; IsACT_Task = True
            except: pass#print("No ACT_Task!!!")
            try: CUT_Task; IsCUT_Task = True
            except: pass#print("No ACT_Task!!!")
            try: Gaze9_Task; IsGaze9_Task = True
            except: pass#print("No Gaze9_Task!!!")
            """Run Analysis"""
            if IsACT_Task:
                ACT_Task.showVideo = False
                ACT_Task.MergeFile()
                ACT_Task.Exec()
            else:
                ACT_Task = Neurobit_Lib_new.ACT_Task(csv_path)  
                ACT_Task.miss_OD = np.nan
                ACT_Task.miss_OS = np.nan
                ACT_Task.NeurobitDx_H = np.nan
                ACT_Task.NeurobitDx_V = np.nan
                ACT_Task.NeurobitDxDev_H = np.nan
                ACT_Task.NeurobitDxDev_V = np.nan
                
            if IsCUT_Task:
                CUT_Task.showVideo = False
                CUT_Task.MergeFile()
                CUT_Task.Exec()
            else:
                CUT_Task = Neurobit_Lib_new.CUT_Task(csv_path)  
                CUT_Task.miss_OD = np.nan
                CUT_Task.miss_OS = np.nan
                CUT_Task.NeurobitDx_H = np.nan
                CUT_Task.NeurobitDx_V = np.nan
                CUT_Task.NeurobitDxDev_H = np.nan
                CUT_Task.NeurobitDxDev_V = np.nan
                
            if IsGaze9_Task:
                Gaze9_Task.showVideo = False
                Gaze9_Task.MergeFile()
                if IsACT_Task: Gaze9_Task.Exec(ACT_Task)   
                else: Gaze9_Task.Exec()
            else:
                Gaze9_Task = Neurobit_Lib_new.Gaze9_Task(csv_path)  
                Gaze9_Task.miss_OD = np.nan
                Gaze9_Task.miss_OS = np.nan
                Gaze9_Task.NeurobitDxDev_H = np.empty([9,2])*np.nan
                Gaze9_Task.NeurobitDxDev_V = np.empty([9,2])*np.nan
                Gaze9_Task.Diff_H = np.empty([9,2])*np.nan
                Gaze9_Task.Diff_V = np.empty([9,2])*np.nan
                
# =============================================================================
#             """Plot Report"""    
#             PDF_Header = sub_head("NeuroSpeed")
#             if IsACT_Task:
#                 Subject_Table   = subject_table(ACT_Task)
#                 Clinic_Table    = clinic_table(ACT_Task)
#                 pdf_path    = os.path.join(ACT_Task.saveReport_path,
#                                            ACT_Task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
#                                            "_OcularMotility.pdf")
#                 pdf         = CreatePDF(pdf_path)
#             elif IsCUT_Task:
#                 Subject_Table   = subject_table(CUT_Task)
#                 Clinic_Table    = clinic_table(CUT_Task)
#                 pdf_path    = os.path.join(CUT_Task.saveReport_path, 
#                                            CUT_Task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
#                                            "_OcularMotility.pdf")
#                 pdf         = CreatePDF(pdf_path)    
#             elif IsGaze9_Task:
#                 Subject_Table   = subject_table(Gaze9_Task)
#                 Clinic_Table    = clinic_table(Gaze9_Task)
#                 pdf_path    = os.path.join(Gaze9_Task.saveReport_path, 
#                                            Gaze9_Task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
#                                            "_OcularMotility.pdf")
#                 pdf         = CreatePDF(pdf_path)
#             
#             if IsACT_Task or IsGaze9_Task or IsCUT_Task:
#                 Sub_Header = sub_head("Clinical relevant data")     
#                 
#                 Element = []
#                 Element.append(PDF_Header)
#                 Element.append(Subject_Table)
#                 Element.append(Sub_Header)
#                 Element.append(Clinic_Table)
#                 if IsACT_Task:
#                     ACTReport(Element, ACT_Task)
#         
#                 Element.append(PageBreak())
#                 if IsCUT_Task:
#                     CUTReport(Element, CUT_Task)
#                     Element.append(PageBreak())
#         
#                 if IsGaze9_Task:
#                     Gaze9Report(Element, Gaze9_Task)
#         
#                 pdf.build(Element)
#                 subprocess.Popen(pdf_path, shell=True)
# =============================================================================
    Subject.GetACT_Save()
# =============================================================================
#     ACT_dx_pd = pd.DataFrame(Subject.ACT_dx) 
# =============================================================================
    ACT_Dx_true_pd = pd.DataFrame.from_dict(Subject.Dx_true, orient='index').transpose() 
# =============================================================================
#     ACT_dx_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx.xlsx'))   
# =============================================================================
    ACT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx_true.xlsx'))  
    
    Subject.GetGaze9_Save()
# =============================================================================
#     Gaze9_dx_pd = pd.DataFrame(Subject.Gaze9_dx) 
# =============================================================================
    Gaze9_Dx_true_pd = pd.DataFrame.from_dict(Subject.Dx_true, orient='index').transpose() 
# =============================================================================
#     Gaze9_dx_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_dx.xlsx'))   
# =============================================================================
    Gaze9_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_Dx_true.xlsx'))
    
    Subject.GetCUT_Save()
# =============================================================================
#     CUT_dx_pd = pd.DataFrame(Subject.CUT_dx) 
# =============================================================================
    CUT_Dx_true_pd = pd.DataFrame.from_dict(Subject.Dx_true, orient='index').transpose() 
# =============================================================================
#     CUT_dx_pd.to_excel(os.path.join(Subject.save_path,'CUT_dx.xlsx'))   
# =============================================================================
    CUT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'CUT_Dx_true.xlsx'))
