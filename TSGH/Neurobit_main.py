# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:19:18 2021

@author: luc40
"""
import glob
import os
import math
import numpy as np
import pandas as pd 
import Neurobit_Lib
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer
from function_PlotReport import main_head, sub_head, con_text, subject_table,\
    clinic_table, diagnose_table, quality_bar, foot1, ACTReport, CreatePDF, Gaze9Report, CUTReport

Fail = []
Subject = Neurobit_Lib.Neurobit()
            
if __name__== '__main__':
    main_path = "E:\\Result\\ALL"
    folderList = Subject.GetFolderPath(main_path)
    for folder in folderList:
        try: del Gaze9_Task
        except: pass
        try: del ACT_Task
        except: pass
        try: del CUT_Task
        except: pass
        csv_files = Subject.GetSubjectFiles(folder)   
        IsACT_Task = False; IsGaze9_Task = False; IsCUT_Task = False;
        for csv_path in csv_files:
            Subject.GetProfile(csv_path)
            print(Subject.Task)            
            if "9 Gaze" in Subject.Task:
                try:
                    Gaze9_Task.session.append(csv_path)
                except:
                    print(csv_path)
                    Gaze9_Task = Neurobit_Lib.Gaze9_Task(csv_path)   
                    Gaze9_Task.session.append(csv_path)
            elif Subject.Task == "13 : Alternate Cover (ACT)":
                try:
                    ACT_Task.session.append(csv_path)
                except:
                    ACT_Task = Neurobit_Lib.ACT_Task(csv_path)  
                    ACT_Task.session.append(csv_path)
            elif ("Cover/Uncover" in Subject.Task or "Calibration" in Subject.Task) and int(Subject.Date) < 20210601:
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
            else:
                print("Having no ["+Subject.Task+"] function!")
        
        try: ACT_Task; IsACT_Task = True
        except: print("No ACT_Task!!!")
        try: CUT_Task; IsCUT_Task = True
        except: print("No CUT_Task!!!")
        try: Gaze9_Task; IsGaze9_Task = True
        except: print("No Gaze9_Task!!!")
        """Run Analysis"""
        if IsACT_Task:
            ACT_Task.showVideo = True
            ACT_Task.MergeFile()
            ACT_Task.Exec()
            #ACT_Task.GetQuality()
        
        if IsCUT_Task:
            CUT_Task.showVideo = True
            CUT_Task.MergeFile()
            CUT_Task.Exec()
            #ACT_Task.GetQuality()
            
        if IsGaze9_Task:
            Gaze9_Task.showVideo = True
            Gaze9_Task.MergeFile()
            try: Gaze9_Task.Exec(ACT_Task)   
            except: Gaze9_Task.Exec()
        
            
        """Plot Report"""
        PDF_Header = main_head("Neurobit")
        if IsACT_Task:
            Subject_Table = subject_table(ACT_Task)
            Clinic_Table = clinic_table(ACT_Task)
            pdf = CreatePDF(os.path.join(ACT_Task.saveReport_path, ACT_Task.FolderName+"_report.pdf"))
        elif IsGaze9_Task:
            Subject_Table = subject_table(Gaze9_Task)
            Clinic_Table = clinic_table(Gaze9_Task)
            pdf = CreatePDF(os.path.join(Gaze9_Task.saveReport_path, Gaze9_Task.FolderName+"_report.pdf"))
        if IsCUT_Task:
            Subject_Table = subject_table(CUT_Task)
            Clinic_Table = clinic_table(CUT_Task)
            pdf = CreatePDF(os.path.join(CUT_Task.saveReport_path, CUT_Task.FolderName+"_report.pdf"))
        
        if IsACT_Task or IsGaze9_Task or IsCUT_Task:
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
                CUTReport(Element, CUT_Task)
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
    #ACT_image_QT_pd = pd.DataFrame({ key:pd.Series(value) for key, value in Subject.ACT_image_QT.items() }) 
    ACT_dx_pd = pd.DataFrame(Subject.ACT_dx) 
    ACT_Dx_true_pd = pd.DataFrame(Subject.Dx_true) 
    #ACT_image_QT_pd.to_excel(os.path.join(Subject.save_path,'ACT_image_QT_new.xlsx'))  
    ACT_dx_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx_TAO.xlsx'))   
    ACT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'ACT_dx_true_TAO.xlsx'))  
    
    Subject.GetGaze9_Save()
    Gaze9_dx_pd = pd.DataFrame(Subject.Gaze9_dx) 
    Gaze9_Dx_true_pd = pd.DataFrame(Subject.Gaze9_Dx_true)
    Gaze9_dx_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_dx_TAO.xlsx'))   
    Gaze9_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'Gaze9_Dx_true_TAO.xlsx'))
    
    Subject.GetCUT_Save()
    CUT_dx_pd = pd.DataFrame(Subject.CUT_dx) 
    CUT_Dx_true_pd = pd.DataFrame(Subject.CUT_Dx_true)
    CUT_dx_pd.to_excel(os.path.join(Subject.save_path,'CUT_dx.xlsx'))   
    CUT_Dx_true_pd.to_excel(os.path.join(Subject.save_path,'CUT_Dx_true.xlsx'))
    
    