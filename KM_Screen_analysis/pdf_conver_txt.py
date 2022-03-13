# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 21:44:24 2022

@author: luc40
"""

# =============================================================================
# pip install pillow
# pip install opencv-python
# pip install fitz
# pip install PyMuPDF
# pip install pytesseract
# =============================================================================
import glob
import numpy as np
from PIL import Image
import pandas as pd 
import os
import pytesseract 
import cv2
import fitz
from pdf2image import convert_from_path
from matplotlib import pyplot as plt

ID       = []
Exam_Date	     = []
Inner_Area_OS	 = []
Outer_Area_OS	 = []
Inner_Area_OD	 = []
Outer_Area_OD	 = []
SIN_RS	 = []
SIN_RL	 = []
SIN_RI	 = []
SIN_OI	 = []
SIN_RM	 = []
SIN_OS	 = []
DX_RS	 = []
DX_RL	 = []
DX_RI	 = []
DX_OI	 = []
DX_RM	 = []
DX_OS	 = []
SIN_D	 = []
SIN_F	 = []
SIN_L	 = []
SIN_LD	 = []
SIN_LU	 = []
SIN_R	 = []
SIN_RD	 = []
SIN_RU	 = []
SIN_U	 = []
DX_D	 = []
DX_F	 = []
DX_L	 = []
DX_LD	 = []
DX_LU	 = []
DX_R	 = []
DX_RD	 = []
DX_RU	 = []
DX_U     = []
def pdf_image(pdf_path,img_path,zoom_x,zoom_y,rotation_angle):
  # 打開PDF文件
  pdf = fitz.open(pdf_path)
  # 逐頁讀取PDF
  for pg in range(0, pdf.pageCount):
    page = pdf[pg]
    # 設置縮放和旋轉系數
    trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
    pm = page.getPixmap(matrix=trans, alpha=False)
    # 開始寫圖像
    pm.writePNG(img_path)
    #pm.writePNG(img_path)
  pdf.close()
def modify_contrast_and_brightness2(img, brightness=0 , contrast=100):
    # 上面做法的問題：有做到對比增強，白的的確更白了。
    # 但沒有實現「黑的更黑」的效果
    import math

    brightness = 0
    contrast = +100 # - 減少對比度/+ 增加對比度

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img = (img - 127.5 * (1 - B)) * k + 127.5 * (1 + B)

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
def mark_region(image_path):
    
    im = cv2.imread(image_path)

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = modify_contrast_and_brightness2(gray)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    ret,thresh = cv2.threshold(blur,180,255,1)
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    plt.figure(figsize=(10,10))
    plt.imshow(thresh)

    line_items_coordinates = []
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        #print(area)
        if y <=500:
            if area > 10000:
                image = cv2.rectangle(im, (x,y), (x+w, y+h), color=(255,0,255), thickness=3)
                line_items_coordinates.append([(x,y), (x+w, y+h)])
        if y <=2100 and y >=1750:
            if area > 10000:
                image = cv2.rectangle(im, (x,y), (x+w, y+h), color=(255,0,255), thickness=3)
                line_items_coordinates.append([(x,y), (x+w, y+h)])
        if y <=2250 and y >2100:
            h = 60; w = int(w/2)
            for i in range(2):
                image = cv2.rectangle(im, (x+w*i,y), (x+w*(i+1), y+h), color=(255,0,255), thickness=3)
                line_items_coordinates.append([(x+w*i,y), (x+w*(i+1), y+h)])
        if y <=2800 and y>2250:
            image = cv2.rectangle(im, (x,y), (x+w, y+h), color=(255,0,255), thickness=3)
            line_items_coordinates.append([(x,y), (x+w, y+h)])
        if y <=4000 and y >2800:
            w = int(w/3)
            h = int((h-75)/3)
            for i in range(3):
                for j in range(3):
                    if i == 0: x_prime = x+w*i+25
                    elif i == 1: x_prime = x+w*i+150
                    elif i == 2: x_prime = x+w*(i+1)-125
                    image = cv2.rectangle(im, (x_prime,y+h*j+25), (x_prime+100,y+h*j+100), color=(255,0,255), thickness=3)
                    line_items_coordinates.append([(x_prime,y+h*j+25), (x_prime+100,y+h*j+100)])
    return image, line_items_coordinates


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pdf_main_path = "E:\\Result\\KMS screen (new)"
img_main_path = "E:\Result\KMS screen image"
pdf_paths = glob.glob(pdf_main_path+"\\*.pdf")
line_items_coordinates = [[(1525, 2900), (1625, 2975)], [(1525, 3166), (1625, 3241)], [(1525, 3432), (1625, 3507)],
                          [(2037, 2900), (2137, 2975)], [(2037, 3166), (2137, 3241)], [(2037, 3432), (2137, 3507)],
                          [(2536, 2900), (2636, 2975)], [(2536, 3166), (2636, 3241)], [(2536, 3432), (2636, 3507)],
                          [(334, 2900), (434, 2975)], [(334, 3164), (434, 3239)], [(334, 3428), (434, 3503)],
                          [(846, 2900), (946, 2975)], [(846, 3164), (946, 3239)], [(846, 3428), (946, 3503)],
                          [(1345, 2900), (1445, 2975)], [(1345, 3164), (1445, 3239)], [(1345, 3428), (1445, 3503)],
                          [(2000, 2628), (2286, 2696)], [(1696, 2628), (1991, 2696)],
                          [(868, 2628), (1163, 2696)], [(573, 2628), (868, 2696)],
                          [(2304, 2389), (2450, 2457)], [(1532, 2394), (1682, 2463)],                             
                          [(1178, 2389), (1328, 2458)], [(409, 2389), (559, 2457)],
                          [(1696, 2170), (1991, 2230)], [(2000, 2170), (2286, 2230)],
                          [(573, 2170), (868, 2230)], [(868, 2170), (1163, 2230)],
                          [(30, 1860), (2830, 2008)], [(85, 192), (435, 279)], [(1296, 107), (1540, 171)]]
for pdf_path in pdf_paths:
    img_path = os.path.join(img_main_path, pdf_path.split("\\")[-1].replace("pdf","png"))
    pages = convert_from_path(pdf_path, 350
                              ,poppler_path=r'C:\Program Files\poppler-0.68.0\bin')
    i = 1
    for page in pages:
        page.save(img_path, "JPEG")
        i = i+1  
      
# =============================================================================
#     image, line_items_coordinates = mark_region(img_path)
# =============================================================================
    
    image = cv2.imread(img_path)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    
    # get co-ordinates to crop the image
    text = []
    for c in line_items_coordinates:
        # cropping image img = image[y0:y1, x0:x1]
        img = image[c[0][1]:c[1][1], c[0][0]:c[1][0]]    
        
        # convert the image to black and white for better OCR
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret,thresh1 = cv2.threshold(gray,170,255,1)
# =============================================================================
#         plt.figure(figsize=(10,10))
#         plt.imshow(thresh1)
# =============================================================================
        # pytesseract image to string to get results
        txt = str(pytesseract.image_to_string(thresh1)).replace("\n","")
        text.append(txt)
        
    ID.append(text[32].split("-")[0])
    Exam_Date.append(text[31])
    Inner_Area_OS.append(text[30].split()[-10])
    Outer_Area_OS.append(text[30].split()[-7])
    Inner_Area_OD.append(text[30].split()[-4])
    Outer_Area_OD.append(text[30].split()[-1])
    SIN_OI.append(text[29])
    SIN_RS.append(text[28])
    SIN_RL.append(text[25])
    SIN_RM.append(text[24])
    SIN_RI.append(text[21])
    SIN_OS.append(text[20])
    
    DX_RS.append(text[27])
    DX_OI.append(text[26])
    DX_RM.append(text[23])
    DX_RL.append(text[22])
    DX_OS.append(text[19])
    DX_RI.append(text[18])
    
    SIN_D.append(text[14])
    SIN_F.append(text[13])
    SIN_L.append(text[10])
    SIN_LD.append(text[11])
    SIN_LU.append(text[9])
    SIN_R.append(text[16])
    SIN_RD.append(text[17])
    SIN_RU.append(text[15])
    SIN_U.append(text[12])
    
    DX_D.append(text[5])
    DX_F.append(text[4])
    DX_L.append(text[1])
    DX_LD.append(text[2])
    DX_LU.append(text[0])
    DX_R.append(text[7])
    DX_RD.append(text[8])
    DX_RU.append(text[6])
    DX_U.append(text[3])

XLS_data = pd.DataFrame({"ID": ID,
            "Exam_Date": Exam_Date,
            "Inner_Area_OS": Inner_Area_OS,
            "Outer_Area_OS": Outer_Area_OS,
            "Inner_Area_OD": Inner_Area_OD,
            "Outer_Area_OD": Outer_Area_OD,
            "SIN_RS": SIN_RS,
            "SIN_RL": SIN_RL,
            "SIN_RI": SIN_RI,
            "SIN_OI": SIN_OI,
            "SIN_RM": SIN_RM,
            "SIN_OS": SIN_OS,
            "DX_RS": DX_RS,
            "DX_RL": DX_RL,
            "DX_RI": DX_RI,
            "DX_OI": DX_OI,
            "DX_RM": DX_RM,
            "DX_OS": DX_OS,
            "SIN_D": SIN_D,
            "SIN_F": SIN_F,
            "SIN_L": SIN_L,
            "SIN_LD": SIN_LD,
            "SIN_LU": SIN_LU,
            "SIN_R": SIN_R,
            "SIN_RD": SIN_RD,
            "SIN_RU": SIN_RU,
            "SIN_U": SIN_U,
            "DX_D": DX_D,
            "DX_F": DX_F,
            "DX_L": DX_L,
            "DX_LD": DX_LD,
            "DX_LU": DX_LU,
            "DX_R": DX_R,
            "DX_RD": DX_RD,
            "DX_RU": DX_RU,
            "DX_U": DX_U})
XLS_data.to_excel(os.path.join("E:\Peggy_analysis\RESULT",'KM_Screen.xlsx'))
