# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 22:21:01 2022

@author: luc40
"""

import os
import cv2
import time
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def GetVideo(csv_path):
    fall = csv_path.replace(".csv",".avi")
    if not os.path.isfile(fall):
        fall = csv_path.replace(".csv",".mp4") 
    return cv2.VideoCapture(fall)
            
class EyeTrackSys:
    def __init__(self, master, csv_path, pic):
        """Initial Setting"""
        my_dpi = 600
        home: int = 0
        visitor: int = 0
        geo_size = '1320x640'
        self.pic = pic
        self.master = master
        self.xy = []
        master.title("Calibration")
        master.geometry(geo_size) 
        plt.clf()
        
        self.csv_path = csv_path
        self.cap = GetVideo(self.csv_path)
        self.cap.set(1,self.pic)
        ret, im = self.cap.read()
        if not ret: 
            self.master.destroy()
            return
        height = im.shape[0]
        width = im.shape[1]
        
        self.fig = plt.figure(figsize=(1280/my_dpi, 480/my_dpi), dpi=my_dpi,frameon=False)
        self.ax1 = self.fig.add_axes([0, -0.15, 1, 1.2])
        self.ax1.imshow(im)
        self.ax1.set_xlim(0,width)
        self.ax1.set_ylim(490,150)        
        self.ax1.axis('off')
        self.ax1.text(20,460, "OD",fontsize=5)
        self.ax1.text(880,460, "OS",fontsize=5)
        self.im1 = FigureCanvasTkAgg(self.fig, master)
        self.im1.get_tk_widget().place(x=20, y=50)        
        self.scat = self.ax1.scatter([],[], s=10, c="b", marker='+',linewidth=.3)
        """Define Cursor"""
        self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        
        """Create Annotation Box"""
        self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
        self.annot.set_visible(False)
        self.fig.canvas.draw()
        
        self.textvar = self.ax1.text(500,140, 
                                "Select center of OD pupil.",
                                fontsize=4, va='center', ha='center')
        self.fig.canvas.callbacks.connect('button_press_event', self)  
        
        self.done_button = tk.Button(self.master, text="Skip!", command=self.done)
        self.done_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
     
    
    def __call__(self, event):
        if event.inaxes is not None:
            x = int(event.xdata) 
            y = int(event.ydata)
           
            self.xy.append([x,y])
            self.annot.xy = (x,y)
            text = "({:d},{:d})".format(x,y)
            self.annot.set_text(text)
            self.annot.set_visible(True)
            try: self.scat.remove() 
            except: pass
            self.scat = self.ax1.scatter(np.array(self.xy)[:,0],np.array(self.xy)[:,1], s=10, c="b", marker='+',linewidth=.3)
            self.fig.canvas.draw()
                
            if len(self.xy) == 0: self.textvar.set_text("Select center of OD pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 1: self.textvar.set_text("Select center of OS pupil."); self.fig.canvas.draw();
            elif len(self.xy) == 2: self.master.destroy()
        else:
            print ('Clicked ouside axes bounds but inside plot window')
    def done(self):
        self.xy.append([np.nan,np.nan])
        self.xy.append([np.nan,np.nan])
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    pic = 10
    my_gui = EyeTrackSys(root, csv_path,pic)
    root.mainloop()
        