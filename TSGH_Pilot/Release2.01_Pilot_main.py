# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:19:18 2021

@author: luc40
"""
import os
import numpy as np
import pandas as pd 
import subprocess
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



main_path = os.getcwd()
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
                
                
                if "9 Gaze Motility Test (9Gaze)" in Subject.Task:
                    try: Gaze9_task.session.append(csv_path)
                    except:
                        Gaze9_task = Gaze9_Task(csv_path)   
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
            
            try: ACT_task; IsACT_Task = True
            except: pass#print("No ACT_Task!!!")
            try: CUT_task; IsCUT_Task = True
            except: pass#print("No ACT_Task!!!")
            try: Gaze9_task; IsGaze9_Task = True
            except: pass#print("No Gaze9_Task!!!")
            """Run Analysis"""
            if IsACT_Task:
                ACT_task.showVideo = True
                ACT_task.MergeFile()
                ACT_task.Exec()
            else:
                ACT_task = ACT_Task(csv_path)  
                ACT_task.miss_OD = np.nan
                ACT_task.miss_OS = np.nan
                ACT_task.NeurobitDx_H = np.nan
                ACT_task.NeurobitDx_V = np.nan
                ACT_task.NeurobitDxDev_H = np.nan
                ACT_task.NeurobitDxDev_V = np.nan
                
            if IsCUT_Task:
                CUT_task.showVideo = True
                CUT_task.MergeFile()
                CUT_task.Exec()
            else:
                CUT_task = CUT_Task(csv_path)  
                CUT_task.miss_OD = np.nan
                CUT_task.miss_OS = np.nan
                CUT_task.NeurobitDx_H = np.nan
                CUT_task.NeurobitDx_V = np.nan
                CUT_task.NeurobitDxDev_H = np.nan
                CUT_task.NeurobitDxDev_V = np.nan
                
            if IsGaze9_Task:
                Gaze9_task.showVideo = True
                Gaze9_task.MergeFile()
                if IsACT_Task: Gaze9_task.Exec(ACT_task)   
                else: Gaze9_task.Exec()
            else:
                Gaze9_task = Gaze9_Task(csv_path)  
                Gaze9_task.miss_OD = np.nan
                Gaze9_task.miss_OS = np.nan
                Gaze9_task.NeurobitDxDev_H = np.empty([9,2])*np.nan
                Gaze9_task.NeurobitDxDev_V = np.empty([9,2])*np.nan
                Gaze9_task.Diff_H = np.empty([9,2])*np.nan
                Gaze9_task.Diff_V = np.empty([9,2])*np.nan
                
            """Plot Report"""    
            PDF_Header = sub_head("NeuroSpeed")
            if IsACT_Task:
                Subject_Table   = subject_table(ACT_task)
                Clinic_Table    = clinic_table(ACT_task)
                pdf_path    = os.path.join(ACT_task.saveReport_path,
                                           ACT_task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
                                           "_OcularMotility.pdf")
                pdf         = CreatePDF(pdf_path)
            elif IsCUT_Task:
                Subject_Table   = subject_table(CUT_task)
                Clinic_Table    = clinic_table(CUT_task)
                pdf_path    = os.path.join(CUT_task.saveReport_path, 
                                           CUT_task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
                                           "_OcularMotility.pdf")
                pdf         = CreatePDF(pdf_path)    
            elif IsGaze9_Task:
                Subject_Table   = subject_table(Gaze9_task)
                Clinic_Table    = clinic_table(Gaze9_task)
                pdf_path    = os.path.join(Gaze9_task.saveReport_path, 
                                           Gaze9_task.FolderName.replace("_","_"+datetime.now().strftime("%H%M%S")+"_")+
                                           "_OcularMotility.pdf")
                pdf         = CreatePDF(pdf_path)
            
            if IsACT_Task or IsGaze9_Task or IsCUT_Task:
                Sub_Header = sub_head("Clinical relevant data")     
                
                Element = []
                Element.append(PDF_Header)
                Element.append(Subject_Table)
                Element.append(Sub_Header)
                Element.append(Clinic_Table)
                if IsACT_Task:
                    ACTReport(Element, ACT_task)
        
                Element.append(PageBreak())
                if IsCUT_Task:
                    CUTReport(Element, CUT_task)
                    Element.append(PageBreak())
        
                if IsGaze9_Task:
                    Gaze9Report(Element, Gaze9_task)
        
                pdf.build(Element)
                subprocess.Popen(pdf_path, shell=True)
    Subject.GetACT_Save()
    ACT_dx_pd = pd.DataFrame(Subject.ACT_dx) 
    ACT_dx_pd.to_excel(os.path.join(Subject.save_path,'Release2.01_ACT_dx.xlsx'))   
    
    Subject.GetGaze9_Save()
    Gaze9_dx_pd = pd.DataFrame(Subject.Gaze9_dx) 
    Gaze9_dx_pd.to_excel(os.path.join(Subject.save_path,'Release2.01_Gaze9_dx.xlsx'))   
    
    Subject.GetCUT_Save()
    CUT_dx_pd = pd.DataFrame(Subject.CUT_dx) 
    CUT_dx_pd.to_excel(os.path.join(Subject.save_path,'Release2.01_CUT_dx.xlsx'))   
