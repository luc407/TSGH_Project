# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 23:19:51 2022

@author: luc40
"""

import os
import tkinter as tk
import pandas as pd
import numpy as np
from Neurobit import Neurobit
from EyeTrack import EyeTrackSys
from Gaze9_Task import Gaze9_Task
from ACT_Task import ACT_Task
from CUT_Task import CUT_Task
from Neurobit import Neurobit

main_path = os.getcwd()
Fail = []
Subject = Neurobit()
Fail = []
Subject = Neurobit()
            
if __name__== '__main__':
    Neurobit.Release_ver = "Release2.02_pilot"
    main_path = "E:\\Result\\"+ Neurobit.Release_ver +"\\Result"
    Subject.DB_path = "E:\\Result\\"+ Neurobit.Release_ver
    folderList = Subject.GetFolderPath(main_path)
    for folder1 in folderList[2:]:        
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
                eye_file = os.path.join(ACT_task.save_path, ACT_task.FolderName+"\\"+ACT_task.task+"_EyePositionCsv.xlsx")
                df = pd.read_excel(eye_file)
                for pic in range(0,len(df)):
                    if np.isnan(df.OD_x[pic]) or np.isnan(df.OS_x[pic]):
                        root = tk.Tk()
                        my_gui = EyeTrackSys(root, csv_path,pic)
                        root.mainloop()
                        if my_gui.xy:
                            df.OD_x[pic] = my_gui.xy[0][0]
                            df.OD_y[pic] = my_gui.xy[0][1]
                            df.OS_x[pic] = my_gui.xy[1][0]
                            df.OS_y[pic] = my_gui.xy[1][1]
                            df.to_excel(os.path.join(ACT_task.save_path, ACT_task.FolderName+"\\"+ACT_task.task+"_EyePositionCsv.xlsx"))
                
            try: CUT_task; IsCUT_Task = True
            except: print("No CUT_Task!!!")
            if IsCUT_Task:
                CUT_task.MergeFile()
                eye_file = os.path.join(CUT_task.save_path, CUT_task.FolderName+"\\"+ CUT_task.task+"_EyePositionCsv.xlsx")
                df = pd.read_excel(eye_file)
                for pic in range(0,len(df)):
                    if np.isnan(df.OD_x[pic]) or np.isnan(df.OS_x[pic]):
                        root = tk.Tk()
                        my_gui = EyeTrackSys(root, csv_path,pic)
                        root.mainloop()
                        if my_gui.xy:
                            df.OD_x[pic] = my_gui.xy[0][0]
                            df.OD_y[pic] = my_gui.xy[0][1]
                            df.OS_x[pic] = my_gui.xy[1][0]
                            df.OS_y[pic] = my_gui.xy[1][1]
                            df.to_excel(os.path.join(CUT_task.save_path, CUT_task.FolderName+"\\"+CUT_task.task+"_EyePositionCsv.xlsx"))
            
            try: Gaze9_task; IsGaze9_Task = True
            except: print("No Gaze9_Task!!!")
            if IsGaze9_Task:
                Gaze9_task.MergeFile()
                eye_file = os.path.join(Gaze9_task.save_path,  Gaze9_task.FolderName+"\\"+Gaze9_task.task+"_EyePositionCsv.xlsx")
                df = pd.read_excel(eye_file)
                for pic in range(0,len(df)):
                    if np.isnan(df.OD_x[pic]) or np.isnan(df.OS_x[pic]):
                        root = tk.Tk()
                        my_gui = EyeTrackSys(root, csv_path,pic)
                        root.mainloop()
                        if my_gui.xy:
                            df.OD_x[pic] = my_gui.xy[0][0]
                            df.OD_y[pic] = my_gui.xy[0][1]
                            df.OS_x[pic] = my_gui.xy[1][0]
                            df.OS_y[pic] = my_gui.xy[1][1]
                            df.to_excel(os.path.join(Gaze9_task.save_path, Gaze9_task.FolderName+"\\"+Gaze9_task.task+"_EyePositionCsv.xlsx"))
                