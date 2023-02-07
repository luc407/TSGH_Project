# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:34:54 2022

@author: luc40
"""

import os
import cv2
import time
import glob
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Cursor
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from moviepy.editor import VideoFileClip, vfx

csv_path = 'E:\\Result\\Release2.01\\Result\\S124740043\\20220808_S124740043\\20220808_111325_S124740043_OcularMotility.csv'

def GetVideo(csv_path):
    fall = csv_path.replace(".csv",".avi")
    if not os.path.isfile(fall):
        fall = csv_path.replace(".csv",".mp4") 
    return cv2.VideoCapture(fall)
            
class CalibSystem:
    def __init__(self, master, csv_path):
        """Initial Setting"""
        my_dpi = 600
        home: int = 0
        visitor: int = 0
        geo_size = '1320x640'
        self.pic_OD = 24
        self.pic_OS = 24
        self.master = master
        self.xy = []
        master.title("Calibration")
        master.geometry(geo_size)        
        
        self.csv_path = csv_path
        self.cap = GetVideo(self.csv_path)
        self.cap.set(1,self.pic_OD)
        ret, im = self.cap.read()
        self.height = im.shape[0]
        self.width = im.shape[1]
        
        
        self.fig = plt.figure(figsize=(1280/my_dpi, 480/my_dpi), dpi=my_dpi,frameon=False)
        self.ax1 = self.fig.add_axes([0, -0.15, 1, 1])
        self.ax1.imshow(im)
        self.ax1.set_xlim(0,self.width)
        self.ax1.set_ylim(490,150)        
        self.ax1.axis('off')
        self.ax1.text(20,440, "OD",fontsize=5)
        self.ax1.text(880,440, "OS",fontsize=5)
        self.im1 = FigureCanvasTkAgg(self.fig, master)
        self.im1.get_tk_widget().place(x=20, y=50)       
        
        self.scat = self.ax1.scatter([],[], s=10, c="b", marker='+',linewidth=.3)
        """Define Cursor"""
        self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        
        """Create Annotation Box"""
        self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
        self.annot.set_visible(False)
        self.fig.canvas.draw()        
        
        self.startCalibrate_button = tk.Button(master, text='Start Calibrate!', command=self.startCalibrate)#.place(relx=0.45, rely=0.9)
        self.startCalibrate_button.place(relx = 0.46, rely = 0.9)
        
        self.Vertical_Flip_button = tk.Button(master, text='Vertical Flip!', command=self.VerFlip)#.place(relx=0.45, rely=0.9)
        self.Vertical_Flip_button.place(relx = 0.38, rely = 0.85)
        
        self.Horizontal_Flip_button = tk.Button(master, text='Horizontal Flip!', command=self.HorzFlip)#.place(relx=0.45, rely=0.9)
        self.Horizontal_Flip_button.place(relx = 0.55, rely = 0.85)
        
        self.pre10_OD_button = tk.Button(master, text="<<", command=self.pre10Frame_OD)#.place(relx=0.35, rely=0.9)
        self.pre10_OD_button.place(relx = 0.2, rely = 0.9)
        
        self.pre_OD_button = tk.Button(master, text="<", command=self.preFrame_OD)#.place(relx=0.4, rely=0.9)
        self.pre_OD_button.place(relx = 0.23, rely = 0.9)
        
        self.next_OD_button = tk.Button(master, text=">", command=self.nextFrame_OD)#.place(relx=0.55, rely=0.9)
        self.next_OD_button.place(relx = 0.25, rely = 0.9)
        
        self.next10_OD_button = tk.Button(master, text=">>", command=self.next10Frame_OD)#.place(relx=0.6, rely=0.9)
        self.next10_OD_button.place(relx = 0.27, rely = 0.9)
                
        self.pre10_OS_button = tk.Button(master, text="<<", command=self.pre10Frame_OS)#.place(relx=0.35, rely=0.9)
        self.pre10_OS_button.place(relx = 0.7, rely = 0.9)
        
        self.pre_OS_button = tk.Button(master, text="<", command=self.preFrame_OS)#.place(relx=0.4, rely=0.9)
        self.pre_OS_button.place(relx = 0.73, rely = 0.9)
        
        self.next_OS_button = tk.Button(master, text=">", command=self.nextFrame_OS)#.place(relx=0.55, rely=0.9)
        self.next_OS_button.place(relx = 0.75, rely = 0.9)
        
        self.next10_OS_button = tk.Button(master, text=">>", command=self.next10Frame_OS)#.place(relx=0.6, rely=0.9)
        self.next10_OS_button.place(relx = 0.77, rely = 0.9)
    
    def __call__(self, event):
        if event.inaxes is not None:
            x = int(event.xdata)            
            
            if len(self.xy) == 0:   y = int(event.ydata); self.cursor = Cursor(self.ax1, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=.3)
            elif len(self.xy) == 1: y = self.xy[0][1]
            elif len(self.xy) == 2: y = self.xy[0][1]; self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            elif len(self.xy) == 3: y = int(event.ydata); self.cursor = Cursor(self.ax1, horizOn=False, vertOn=True, useblit=True, color='r', linewidth=.3)
            elif len(self.xy) == 4: y = self.xy[3][1]
            elif len(self.xy) == 5: y = self.xy[3][1]; self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
            else: return
           
            self.xy.append([x,y])
            self.annot.xy = (x,y)
            text = "({:d},{:d})".format(x,y)
            self.annot.set_text(text)
            self.annot.set_visible(True)
            try: self.scat.remove() 
            except: pass
            self.scat = self.ax1.scatter(np.array(self.xy)[:,0],np.array(self.xy)[:,1], s=10, c="b", marker='+',linewidth=.3)
            self.fig.canvas.draw()
            
            if len(self.xy)==6:
                self.done_button = tk.Button(self.master, text="Done!", command=self.done)
                self.done_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
            else:
                try: self.done_button.destroy()
                except: pass
                
            if len(self.xy) == 0: self.textvar.set_text("Select center of OD pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 1: self.textvar.set_text("Select right margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 2: self.textvar.set_text("Select left margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 3: self.textvar.set_text("Select center of OS pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 4: self.textvar.set_text("Select right margin of OS iris."); self.fig.canvas.draw()
            elif len(self.xy) == 5: self.textvar.set_text("Select left margin of OS iris."); self.fig.canvas.draw()            
            elif len(self.xy) == 6: self.textvar.set_text("Well done!."); self.fig.canvas.draw()
            elif len(self.xy) > 6: self.textvar.set_text("Too many point!."); self.fig.canvas.draw()
        else:
            print ('Clicked ouside axes bounds but inside plot window')
            
    def popBackXY(self):  
        if len(self.xy)>0:
            self.xy.pop()   
            self.scat.remove()
            if len(self.xy)>0: self.scat = self.ax1.scatter(np.array(self.xy)[:,0],np.array(self.xy)[:,1], s=10, c="b", marker='+',linewidth=.3)
            self.fig.canvas.draw()
            
            if len(self.xy)==6:
                self.done_button = tk.Button(self.master, text="Done!", command=self.done)
                self.done_button.pack(side=tk.LEFT, anchor=tk.S, pady=50)
            else:
                try: self.done_button.destroy()
                except: pass
            
            if len(self.xy) == 0: self.textvar.set_text("Select center of OD pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 1: self.textvar.set_text("Select right margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 2: self.textvar.set_text("Select left margin of OD iris."); self.fig.canvas.draw()
            elif len(self.xy) == 3: self.textvar.set_text("Select center of OS pupil."); self.fig.canvas.draw()
            elif len(self.xy) == 4: self.textvar.set_text("Select right margin of OS iris."); self.fig.canvas.draw()
            elif len(self.xy) == 5: self.textvar.set_text("Select left margin of OS iris."); self.fig.canvas.draw()            
            elif len(self.xy) == 6: self.textvar.set_text("Well done!."); self.fig.canvas.draw()
            elif len(self.xy) > 6: self.textvar.set_text("Too many point!."); self.fig.canvas.draw()
    
    def done(self):
        self.OD_WTW = abs(self.xy[1][0]-self.xy[2][0])
        self.OS_WTW = abs(self.xy[4][0]-self.xy[5][0])
        self.master.destroy()
    
    def nextFrame_OD(self):
        self.pic_OD+=30
        if self.pic_OD < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,self.pic_OD)
            ret, im = self.cap.read()
            self.ax1.imshow(im[:,0:int(self.width/2)])
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OD-=30
    
    def next10Frame_OD(self):
        self.pic_OD+=150
        if self.pic_OD < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,self.pic_OD)
            ret, im = self.cap.read()
            self.ax1.imshow(im[:,0:int(self.width/2)])
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OD-=150
            
    def preFrame_OD(self):
        self.pic_OD-=30
        if self.pic_OD >= 0 :
            self.cap.set(1,self.pic_OD)
            ret, im = self.cap.read()
            self.ax1.imshow(im[:,0:int(self.width/2)])
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)            
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OD+=30
    
    def pre10Frame_OD(self):
        self.pic_OD-=150
        if self.pic_OD >= 0 :
            self.cap.set(1,self.pic_OD)
            ret, im = self.cap.read()
            self.ax1.imshow(im[:,0:int(self.width/2)])
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OD+=150
    
    def nextFrame_OS(self):
        self.pic_OS+=30
        if self.pic_OS < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,self.pic_OS)
            ret, im = self.cap.read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im[:,0:int(self.width/2),3] = 0
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OS-=30
    
    def next10Frame_OS(self):
        self.pic_OS+=150
        if self.pic_OS < int(cv2.VideoCapture.get(self.cap, int(cv2.CAP_PROP_FRAME_COUNT))):
            self.cap.set(1,self.pic_OS)
            ret, im = self.cap.read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im[:,0:int(self.width/2),3] = 0
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OS-=150
            
    def preFrame_OS(self):
        self.pic_OS-=30
        if self.pic_OS >= 0 :
            self.cap.set(1,self.pic_OS)
            ret, im = self.cap.read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im[:,0:int(self.width/2),3] = 0
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)            
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OS+=30
    
    def pre10Frame_OS(self):
        self.pic_OS-=150
        if self.pic_OS >= 0 :
            self.cap.set(1,self.pic_OS)
            ret, im = self.cap.read()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
            im[:,0:int(self.width/2),3] = 0
            self.ax1.imshow(im)
            self.ax1.axis('off')
            self.im1 = FigureCanvasTkAgg(self.fig, self.master)
            self.im1.get_tk_widget().place(x=20, y=50)
            
            
            """Create Annotation Box"""
            self.annot = self.ax1.annotate("", xy=(0,0), xytext=(-5,5), textcoords='offset points', size = 3)
            self.annot.set_visible(False)
            self.fig.canvas.draw()
            """Define Cursor"""
            self.cursor = Cursor(self.ax1, horizOn=True, vertOn=True, useblit=True, color='r', linewidth=.3)
        else:
            self.pic_OS+=150
    
        
    def startCalibrate(self):
        self.startCalibrate_button.destroy()
        self.pre10_OD_button.destroy()
        self.pre_OD_button.destroy()
        self.next_OD_button.destroy()
        self.next10_OD_button.destroy()
        self.pre10_OS_button.destroy()
        self.pre_OS_button.destroy()
        self.next_OS_button.destroy()
        self.next10_OS_button.destroy()
        self.Vertical_Flip_button.destroy()
        self.Horizontal_Flip_button.destroy()
        
        self.textvar = self.ax1.text(500,120, 
                                    "Select center of OD pupil.",
                                    fontsize=4, va='center', ha='center')
        self.fig.canvas.draw()
        
        self.popBack_button = tk.Button(self.master, text="Back", command=self.popBackXY)
        self.popBack_button.pack(side=tk.LEFT, padx=400, anchor=tk.S, pady=50) 
        
        self.fig.canvas.callbacks.connect('button_press_event', self)       
    
    def VerFlip(self):
        path_tmp = os.path.abspath(os.path.join(self.csv_path, os.path.pardir))
        video_list = glob.glob(path_tmp+"\*.mp4")
        for mp4 in video_list:
            print(mp4)
            tmp_mp4 = os.path.join(path_tmp,"test.mp4")
            video = VideoFileClip(mp4)
            video = video.fx(vfx.mirror_y)
            video.write_videofile(tmp_mp4)
            video_tmp = VideoFileClip(tmp_mp4)
            video_tmp.write_videofile(mp4)
            os.remove(tmp_mp4)
        self.cap = GetVideo(self.csv_path)
        self.cap.set(1,self.pic_OD)
        ret, im = self.cap.read()
        self.ax1.imshow(im)
        self.ax1.axis('off')
        self.im1 = FigureCanvasTkAgg(self.fig, self.master)
        self.im1.get_tk_widget().place(x=20, y=50)
    
    def HorzFlip(self):
        path_tmp = os.path.abspath(os.path.join(self.csv_path, os.path.pardir))
        video_list = glob.glob(path_tmp+"\*.mp4")
        for mp4 in video_list:
            print(mp4)
            tmp_mp4 = os.path.join(path_tmp,"test.mp4")
            video = VideoFileClip(mp4)
            video = video.fx(vfx.mirror_x)
            video.write_videofile(tmp_mp4)
            video_tmp = VideoFileClip(tmp_mp4)
            video_tmp.write_videofile(mp4)
            os.remove(tmp_mp4)
        self.cap = GetVideo(self.csv_path)
        self.cap.set(1,self.pic_OD)
        ret, im = self.cap.read()
        self.ax1.imshow(im)
        self.ax1.axis('off')
        self.im1 = FigureCanvasTkAgg(self.fig, self.master)
        self.im1.get_tk_widget().place(x=20, y=50)
        
if __name__ == "__main__":
    root = tk.Tk()
    my_gui = CalibSystem(root, csv_path)
    root.mainloop()
    