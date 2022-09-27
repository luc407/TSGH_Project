# -*- coding: utf-8 -*-
"""
Created on Thu May 19 02:26:10 2022

@author: luc40
"""
import os
import Neurobit_Lib_pilot
import tkinter as tk
from calibration import CalibSystem



main_path = os.getcwd()
Fail = []
Subject = Neurobit_Lib_pilot.Neurobit()

if __name__== '__main__':    
    main_path = "E:\\Result\\NS01_1_tsgh_flight\\Release2.02\\Result"
    folderList = Subject.GetFolderPath(main_path)
    for folder1 in folderList[1:]:
        sub_folderList = Subject.GetFolderPath(folder1)
        for folder2 in sub_folderList:
            IsCalibrated = False
            csv_files = Subject.GetSubjectFiles(folder2)        
            for csv_path in csv_files:
                write_path = os.path.join(Subject.save_csv_path,csv_path.split("\\")[-2]+"\\cal_param.txt")
                if not os.path.isdir(write_path.replace("\\cal_param.txt","")):
                    os.makedirs(write_path.replace("\\cal_param.txt",""))
                if not IsCalibrated:
                    root = tk.Tk()
                    my_gui = CalibSystem(root, csv_path)
                    root.mainloop()
                    IsCalibrated = True
                    with open(write_path, 'w') as f:
                        f.write(str(my_gui.OD_WTW) + '\n')
                        f.write(str(my_gui.OS_WTW) + '\n')
                        f.write(str(my_gui.xy[0][0]) + " " + str(my_gui.xy[0][1]) + '\n')
                        f.write(str(my_gui.xy[3][0]) + " " + str(my_gui.xy[3][1]) + '\n')
                else:
                    break
