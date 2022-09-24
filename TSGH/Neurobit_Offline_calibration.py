# -*- coding: utf-8 -*-
"""
Created on Thu May 19 02:26:10 2022

@author: luc40
"""
import os
import Neurobit_Lib
import tkinter as tk
from calibration import CalibSystem



main_path = os.getcwd()
Fail = []
Subject = Neurobit_Lib.Neurobit()
            
if __name__== '__main__':    
    main_path = "E:\Result\ALL"
    folderList = Subject.GetFolderPath(main_path)
    for folder in folderList:
        IsCalibrated = False
        csv_files = Subject.GetSubjectFiles(folder)        
        for csv_path in csv_files:
            Subject.GetProfile(csv_path)
            write_path = os.path.join(Subject.save_csv_path,csv_path.split("\\")[-2]+"\\cal_param.txt")
            try: os.makedirs(os.path.join(Subject.save_csv_path,csv_path.split("\\")[-2]))
            except: pass
            if not IsCalibrated:
                if(("Cover/Uncover" in Subject.Task and int(Subject.Date) < 20210601) or
                ("Calibration" in Subject.Task and int(Subject.Date) < 20210601) or
                (Subject.Task == "13 : Alternate Cover (ACT)") or
                (csv_path == csv_files[-1])):
                    root = tk.Tk()
                    my_gui = CalibSystem(root, csv_path)
                    root.mainloop()
                    IsCalibrated = True
                    with open(write_path, 'w') as f:
                        f.write(str(my_gui.OD_WTW) + '\n')
                        f.write(str(my_gui.OS_WTW) + '\n')
                        f.write(str(my_gui.xy[0][0]) + " " + str(my_gui.xy[0][1]) + '\n')
                        f.write(str(my_gui.xy[3][0]) + " " + str(my_gui.xy[3][1]) + '\n')
            elif IsCalibrated:
                break

if __name__== '__main__':    
    main_path = "E:\Result\Release2.01\Result"
    folderList = Subject.GetFolderPath(main_path)
    for folder1 in folderList[12:]:
        sub_folderList = Subject.GetFolderPath(folder1)
        for folder2 in sub_folderList:
            IsCalibrated = False
            csv_files = Subject.GetSubjectFiles(folder2)        
            for csv_path in csv_files:
                write_path = os.path.join(Subject.save_csv_path,csv_path.split("\\")[-2]+"\\cal_param.txt")
                if not IsCalibrated:
                    root = tk.Tk()
                    my_gui = CalibSystem(root, csv_path)
                    root.mainloop()
                    IsCalibrated = True
                    with open(write_path, 'w') as f:
                        f.write(str(my_gui.OD_WTW) + '\n')
                        f.write(str(my_gui.OS_WTW) + '\n')
                else:
                    break
