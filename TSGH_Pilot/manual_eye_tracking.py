# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:30:15 2022

@author: luc40
"""
import os
import Neurobit_Lib_pilot
import tkinter as tk
import pandas as pd
from calibration import EyeTrackSystem



main_path = os.getcwd()
Fail = []
Subject = Neurobit_Lib_pilot.Neurobit()

if __name__== '__main__':    
    main_path = "E:\Result\Release2.01_pilot\Result"
    folderList = Subject.GetFolderPath(main_path)
    for folder1 in folderList[0:1]:
        sub_folderList = Subject.GetFolderPath(folder1)
        for folder2 in sub_folderList:
            IsCalibrated = False
            csv_files = Subject.GetSubjectFiles(folder2)        
            for csv_path in csv_files:
                Subject.FolderName = csv_path.split("\\")[-2]
                Subject.GetProfile(csv_path)
                if "9 Gaze Motility Test (9Gaze)" in Subject.Task:
                    write_path = os.path.join(Subject.save_csv_path,csv_path.split("\\")[-2]+"\\9_Gaze_EyePositionCsv.xlsx")
                    if not IsCalibrated:
                        root = tk.Tk()
                        my_gui = EyeTrackSystem(root, csv_path)
                        root.mainloop()
                        IsCalibrated = True
                        EyePositionCsv = pd.DataFrame({"OD_x":my_gui.OD[0,:],
                                                       "OD_y":my_gui.OD[1,:],
                                                       "OD_p":my_gui.OD[2,:],
                                                       "OS_x":my_gui.OS[0,:],
                                                       "OS_y":my_gui.OS[1,:],
                                                       "OS_p":my_gui.OS[2,:]})
                        EyePositionCsv.to_excel(write_path)  
                    else:
                        break
