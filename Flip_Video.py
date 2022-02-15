# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 01:07:07 2021

@author: luc40
"""
import os
import glob
import pygame
from moviepy.editor import VideoFileClip, vfx


if __name__== '__main__':
    main_path = "E:\\Result\\New20211129"
    with open("E:\\Result\\Horizontal.txt") as H:
        folders = H.readlines()        
        for folder in folders[1:]:            
            path_tmp = os.path.join(main_path,folder.replace("\n",""))
            video_list = glob.glob(path_tmp+"\*.mp4")
            print(video_list)
            for mp4 in video_list:
                print(mp4)
                tmp_mp4 = os.path.join(path_tmp,"test.mp4")
                video = VideoFileClip(mp4)
                video = video.fx(vfx.mirror_x)
                video.write_videofile(tmp_mp4)
                video_tmp = VideoFileClip(tmp_mp4)
                video_tmp.write_videofile(mp4)
                os.remove(tmp_mp4)
                
    with open("E:\\Result\\Horizontal_Vertical.txt") as H_V:
        folders = H_V.readlines()        
        for folder in folders:
            path_tmp = os.path.join(main_path,folder.replace("\n",""))
            video_list = glob.glob(path_tmp+"\*.mp4")
            for mp4 in video_list:
                tmp_mp4 = os.path.join(path_tmp,"test.mp4")
                video = VideoFileClip(mp4)
                video = video.fx(vfx.mirror_x).fx(vfx.mirror_y)
                video.write_videofile(tmp_mp4)
                video_tmp = VideoFileClip(tmp_mp4)
                video_tmp.write_videofile(mp4)

            
        