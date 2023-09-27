# -*- coding: utf-8 -*-
"""
Created on Thu May 19 02:26:10 2022

@author: luc40
"""
import os
from Neurobit import Neurobit
import tkinter as tk
from calibration import CalibSystem



main_path = os.getcwd()
Fail = []
Subject = Neurobit()

if __name__== '__main__':    
    Neurobit.Release_ver = "Release2.01_pilot"
    main_path = "E:\\Result\\"+ Neurobit.Release_ver +"\\Result"
    main_path = "E:\Result\TSGH_G_Project_0810_11"
    save_path = os.getcwd()+"\\RESULT\\Calibration"
    folderList = Subject.GetFolderPath(main_path)
    for folder1 in folderList[7:8]:
        sub_folderList = Subject.GetFolderPath(folder1)
        for folder2 in sub_folderList:
            IsCalibrated = False
            csv_files = Subject.GetSubjectFiles(folder2)        
            for csv_path in csv_files:
                write_path = os.path.join(save_path,csv_path.split("\\")[-2]+"_cal_param.txt")
                if not os.path.isdir(save_path):
                    os.makedirs(save_path)
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
