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
    def __init__(self, csv_path, pic, *args):
        """Initial Setting"""        
        plt.clf()
        my_dpi = 600
        home: int = 0
        visitor: int = 0
        geo_size = '1320x640+0+50'
        self.pic = pic
        self.master = tk.Toplevel()
        self.xy = []
        self.master.title("Calibration")
        self.master.geometry(geo_size)        
        
        self.csv_path = csv_path
        self.cap = GetVideo(self.csv_path)
        self.cap.set(1,self.pic)
        ret, im = self.cap.read()                  
        if ret:
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
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)        
            self.scat = self.ax1.scatter([],[], s=10, c="b", marker='+',linewidth=.3)
        
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            
            """Get Exrta Input"""
            i = 0
            for arg in args:            
                if i == 0:
                    if arg.any():
                        self.Extra = True
                        self.scat = self.ax1.scatter(int(arg[0]),int(arg[1]), s=10, c="b", marker='+',linewidth=.3)
                        self.fig.canvas.draw()
                elif i == 1:
                    self.Extra = True
                    self.trial = arg
                i+=1
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            self.fig.canvas.callbacks.connect('button_press_event', self)  
            self.done_button = tk.Button(self.master, text="Skip!", command=self.done)
            self.done_button.pack(side=tk.LEFT, padx=258, pady=50, anchor=tk.S)
            
            if self.Extra:
                self.pre10Frame_button = tk.Button(self.master, text="<<", command=self.pre24Frame)#.place(relx=0.35, rely=0.9)
                self.pre10Frame_button.pack(side = tk.LEFT, anchor=tk.S, pady=50)
                
                self.preFrame_button = tk.Button(self.master, text="<", command=self.pre12Frame)#.place(relx=0.4, rely=0.9)
                self.preFrame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)        
                
                self.nextFrame_button = tk.Button(self.master, text=">", command=self.next12Frame)#.place(relx=0.55, rely=0.9)
                self.nextFrame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
                
                self.next10Frame_button = tk.Button(self.master, text=">>", command=self.next24Frame)#.place(relx=0.6, rely=0.9)
                self.next10Frame_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)   
                self.textvar = self.ax1.text(500,140, 
                                            "Select center of pupil. "+self.trial,
                                            fontsize=4, va='center', ha='center')
            else:
                self.textvar = self.ax1.text(500,140, 
                                            "Select center of OD pupil.",
                                            fontsize=4, va='center', ha='center')
        else:
            self.master.quit()
            self.master.destroy()  
                
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
            
            if not self.Extra:
                if len(self.xy) == 0: self.textvar.set_text("Select center of OD pupil."); self.fig.canvas.draw()
                elif len(self.xy) == 1: self.textvar.set_text("Select center of OS pupil."); self.fig.canvas.draw();
                elif len(self.xy) == 2: self.master.quit(); self.master.destroy()
            else:
                if len(self.xy) == 1: self.master.quit(); self.master.destroy()
        else:
            print ('Clicked ouside axes bounds but inside plot window')
    def done(self):
        plt.clf()
        self.xy.append([np.nan,np.nan])
        self.xy.append([np.nan,np.nan])
        self.master.quit()
        self.master.destroy()
    
    def next12Frame(self):
        self.pic+=6
        if self.pic < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,self.pic)
            ret, im = self.cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "OD",fontsize=5)
            self.ax1.text(880,460, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic-=30
    
    def next24Frame(self):
        self.pic+=12
        if self.pic < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,self.pic)
            ret, im = self.cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "OD",fontsize=5)
            self.ax1.text(880,460, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic-=150
            
    def pre12Frame(self):
        self.pic-=6
        if self.pic >= 0 :
            self.cap.set(1,self.pic)
            ret, im = self.cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "OD",fontsize=5)
            self.ax1.text(880,460, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)            
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic+=30
    
    def pre24Frame(self):
        self.pic-=12
        if self.pic >= 0 :
            self.cap.set(1,self.pic)
            ret, im = self.cap.read()
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.ax1.text(20,460, "OD",fontsize=5)
            self.ax1.text(880,460, "OS",fontsize=5)
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic+=150
if __name__ == "__main__":
    root = tk.Tk()
    pic = 10
    my_gui = EyeTrackSys(root, csv_path,pic)
    root.mainloop()
        