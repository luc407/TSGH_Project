# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:32:28 2021

@author: luc40
"""

# ReportLab imports
import numpy as np
import Neurobit as nb
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import BaseDocTemplate, Image, Paragraph, Table, TableStyle, PageBreak, \
    Frame, PageTemplate, NextPageTemplate,Spacer 
from reportlab.graphics import renderPDF, renderPM

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont('TSans', 'TaipeiSansTCBeta-Regular.ttf'))
pdfmetrics.registerFont(TTFont('TSans_bold', 'TaipeiSansTCBeta-Bold.ttf'))

TABLE_WIDTH = 7.2   # inch
GAZE_TABLE_WIDTH = 3.6   # inch
H_MARGIN = 0.5 * inch
T_MARGIN = 30
B_MARGIN = 18
P_MARGIN = 10

def main_head(headtext):
    Style=getSampleStyleSheet()
    bt = Style['Normal']    #字體的樣式
    bt.fontName='TSans_bold'     #使用的字體
    bt.fontSize=18          #字號
    bt.wordWrap = 'Normal'     #該屬性支持自動換行，'CJK'是中文模式換行，用於英文中會截斷單詞造成閱讀困難，可改爲'Normal'
    bt.spaceAfter= 16
    return Paragraph(headtext,bt)

def sub_head(headtext):
    Style=getSampleStyleSheet()
    bt = Style['Normal']    #字體的樣式
    bt.fontName='TSans_bold'     #使用的字體
    bt.fontSize=14          #字號
    bt.wordWrap = 'Normal'     #該屬性支持自動換行，'CJK'是中文模式換行，用於英文中會截斷單詞造成閱讀困難，可改爲'Normal'
    bt.spaceBefore= 10
    bt.spaceAfter= 10
    return Paragraph(headtext,bt)

def con_text(headtext):
    Style=getSampleStyleSheet()
    bt = Style['Normal']    #字體的樣式
    bt.fontName='TSans'     #使用的字體
    bt.fontSize=10         #字號
    bt.wordWrap = 'Normal'     #該屬性支持自動換行，'CJK'是中文模式換行，用於英文中會截斷單詞造成閱讀困難，可改爲'Normal'
    bt.spaceBefore= 10
    bt.spaceAfter= 10
    return Paragraph(headtext,bt)

def subject_table(Subject):
    if Subject.Profile_ind:
        try:
            data = [['Patient ID: ' + Subject.ID,       'Date of Birth: ' + Subject.DoB,    'Exam Date: ' + Subject.Date,   'Doctor: ' + Subject.Doctor],
                    ['Patient Name: ' + Subject.Name,   'Gender: ' + Subject.Gender,        'Age: ' + Subject.Age,          'Height: ' + str(Subject.Height)]
            ]   
        except:
            data = [['Patient ID: ' + Subject.ID,       'Date of Birth: ',    'Exam Date: ',   'Doctor: '],
                    ['Patient Name: ',   'Gender: ' ,        'Age: ' ,          'Height: ']
            ]   
    else:
        data = [['Patient ID: ' + Subject.ID,       'Date of Birth: ' + str(Subject.Profile_ind),    'Exam Date: ' + Subject.Date,   'Doctor: ' + Subject.Doctor],
                ['Patient Name: ' + str(Subject.Profile_ind),   'Gender: ' + str(Subject.Profile_ind),        'Age: ' + str(Subject.Profile_ind),          'Height: ' + str(Subject.Profile_ind)]
        ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'TSans'),    # 字體
        ('FONTSIZE', (0, 0), (-1, 0), 10),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),        # 對齊
        ('VALIGN', (-1, 0), (-2, 0), 'MIDDLE'),     # 對齊
        
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),   # 設置表格框線爲grey色，線寬爲0.5
    ]    
    colWidths = (TABLE_WIDTH / len(data[0])) * inch     # 每列的寬度
    component_table = Table(dis_list,colWidths = colWidths, style=style)    
    return component_table

def clinic_table(Subject):
    if Subject.Profile_ind:
        try:
            data = [['Hx: ' + Subject.Dx,'','','','','','','','','',''],
                    ['',    'VAsc',          'VAcc',        'Auto-Ref',     'pupil',            'WTW',          'AXL',          'Hertel',           '',                 'PD',       'Stereo'],
                    ['OD',  Subject.VA_OD, Subject.BCVA_OD, Subject.Ref_OD,  Subject.pupil_OD,  Subject.WTW_OD, Subject.AL_OD, Subject.Hertal_OD,  Subject.Hertal_Len, Subject.PD, Subject.Stereo],
                    ['OS',  Subject.VA_OS, Subject.BCVA_OS, Subject.Ref_OS,  Subject.pupil_OS,  Subject.WTW_OS, Subject.AL_OS, Subject.Hertal_OS,  '',                 '',         ''],
            ]     
        except:
            data = [['Hx: ' ,'','','','','','','','','',''],
                    ['',    'VAsc',          'VAcc',        'Auto-Ref',     'pupil',            'WTW',          'AXL',          'Hertel',           '',                 'PD',       'Stereo'],
                    ['OD', '', '', '', '', '', '', '', '', '', ''],
                    ['OS', '', '', '', '', '', '', '',  '',                 '',         ''],
            ]  
    else:
        data = [['Hx: ' + str(Subject.Profile_ind),'','','','','','','','','',''],
                ['',    'VAsc',          'VAcc',        'Auto-Ref',     'pupil',            'WTW',          'AXL',          'Hertel',           '',                 'PD',       'Stereo'],
                ['OD',  '', '', '', '', '', '', '', '', '', ''],
                ['OS',  '', '', '', '', '', '', '', '', '', ''],
        ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'TSans'),    # 字體
        ('FONTSIZE', (0, 0), (-1, 0), 10),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, 0), 'LEFT'),        # 對齊
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('SPAN',(0,0),(-1,0)),
        ('SPAN',(7,1),(8,1)),
        ('SPAN',(8,2),(8,3)),
        ('SPAN',(9,2),(9,3)),
        ('SPAN',(10,2),(10,3)),
        
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),   # 設置表格框線爲grey色，線寬爲0.5
    ]    
    colWidths = (TABLE_WIDTH / len(data[0])) * inch     # 每列的寬度
    component_table = Table(dis_list,colWidths = colWidths, style=style)    
    return component_table
def diagnose_table(OLD_ACT_Task):
# =============================================================================
#     data=[['','','','','',''],
#           ['','','','','',''],
#           ['','',OLD_ACT_Task.NeurobitDx_H,'\n'+str(np.round(OLD_ACT_Task.NeurobitDxDev_H,1))+' PD','',''],
#           ['','',OLD_ACT_Task.NeurobitDx_V,'\n'+str(np.round(OLD_ACT_Task.NeurobitDxDev_V,1))+' PD','',''],
#           ['','','','','',''],
#           ['','','','','','']]   
#     dis_list = []
#     for x in data:
#         dis_list.append(x)
#     style = [
#         ('FONTNAME', (0, 0), (-1, -1), 'TSans'),    # 字體
#         ('FONTSIZE', (0, 0), (-1, -1), 10),          # 字體大小
#         
#         ('ALIGN', (0, 0), (-1, -1), 'CENTER'),        # 對齊
#         ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
#         
#         ('BOX', (0, 0), (1, 1), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (0, 2), (1, 3), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (0, 4), (1, 5), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (2, 0), (3, 1), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (2, 2), (3, 3), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (2, 4), (3, 5), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (4, 0), (5, 1), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (4, 2), (5, 3), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#         ('BOX', (4, 4), (5, 5), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
#     ]    
# =============================================================================
    data=[[OLD_ACT_Task.NeurobitDx_H,str(np.round(OLD_ACT_Task.NeurobitDxDev_H,1))+' PD'],
          [OLD_ACT_Task.NeurobitDx_V,str(np.round(OLD_ACT_Task.NeurobitDxDev_V,1))+' PD']]   
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'TSans'),    # 字體
        ('FONTSIZE', (0, 0), (-1, -1), 14),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('BOX', (0, 0), (-1, -1), 1.5, colors.black),   # 設置表格框線爲grey色，線寬爲1
    ]
    colWidths = (GAZE_TABLE_WIDTH / len(data[0])) * inch     # 每列的寬度
    component_table = Table(dis_list, 
                            colWidths=colWidths, 
                            rowHeights=colWidths/2,
                            style=style)    
    return component_table
def quality_bar(OD, OS):
    miss_OD = round(np.count_nonzero(np.isnan(OD[0]))/len(OD[0]),2)
    miss_OS = round(np.count_nonzero(np.isnan(OS[0]))/len(OS[0]),2)

    data=[['Missing Point: ','OD: ', str(int(miss_OD*100))+'%','OS: ', str(int(miss_OS*100))+'%']
          ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    
    if 1-miss_OD>=0.9:
        OD_color = colors.lime
    elif 1-miss_OD>=0.7:
        OD_color = colors.yellow
    else:
        OD_color = colors.red
    if 1-miss_OS>=0.9:
        OS_color = colors.lime
    elif 1-miss_OS>=0.7:
        OS_color = colors.yellow
    else:
        OS_color = colors.red
        
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'TSans'),    # 字體
        ('FONTSIZE', (0, 0), (-1, -1), 8),          # 字體大小
        
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('BOX', (0, 0), (-1, -1), 1, colors.grey),   # 設置表格框線爲grey色，線寬爲1
        ('BACKGROUND', (2, 0), (2, 0), OD_color),
        ('BACKGROUND', (4, 0), (4, 0), OS_color)
    ]    
    component_table = Table(dis_list, style=style, rowHeights=10)    
    return component_table
def EyeTrackImage(file_path):
    im = Image(file_path, width=7.2 *inch, height=2.5 *inch)
    im.hAlign = 'CENTER'
    return im
def ActEyeImage(file_path):
    im = Image(file_path, width=3 *inch, height=4.4*inch)
    im.hAlign = 'RIGHT'
    return im
def CutEyeImage(file_path):
    im = Image(file_path, width=3 *inch, height=5*inch)
    im.hAlign = 'RIGHT'
    return im
def Gaze9EyeImage(file_path):
    im = Image(file_path, width=7.2 *inch, height=2.5*inch)
    im.hAlign = 'CENTER'
    return im
def Gaze9EyeMesh(file_path):
    im = Image(file_path, width=6/1.2 *inch, height=3/1.2 *inch)
    im.hAlign = 'CENTER'
    return im
def QRCodeImage(file_path):
    im = Image(file_path, width=2 *inch, height=2*inch)
    im.hAlign = 'LEFT'
    return im
def foot1(canvas, can):
    page = "Page {}".format(can.page)
    canvas.saveState()
    canvas.setFont('TSans', 10)
    canvas.setFillColorRGB(.5,.5,.5)
    canvas.drawString((can.width+len(page)*7) / 2, 0.5 * inch, page)
    canvas.restoreState()
def CreatePDF(file_path):
    can = BaseDocTemplate(file_path, 
                          pagesize=A4,
                          rightMargin=H_MARGIN,leftMargin=H_MARGIN,
                          topMargin=T_MARGIN, bottomMargin=B_MARGIN)     
    frameT = Frame(can.leftMargin, can.bottomMargin, can.width, can.height, id='normal')
    can.addPageTemplates([PageTemplate(id='OneCol', frames=frameT, onPage=foot1),])
    return can
def ACTReport(Element, CUT_Task):
    sub_head2 = sub_head("ACT Dynamic Eyeposition Tracking")
    sub_head3 = sub_head("Ocular Alignment Table")
    text1 = con_text("Alternated Cover Test Sequence in Primary Position")
    Quality_Bar = quality_bar(CUT_Task.OD, CUT_Task.OS)
    gaze_table = diagnose_table(CUT_Task)
    
    Element.append(sub_head2)
    im1 = EyeTrackImage(CUT_Task.saveImage_path+"\\DrawEyeTrack.png")
    Element.append(im1)
    Element.append(Quality_Bar)
    Element.append(sub_head3)
    Element.append(text1)    
    im2 = ActEyeImage(CUT_Task.saveImage_path+"\\DrawEyeFig.png")
    im3 = QRCodeImage(CUT_Task.saveImage_path+"\\QR_code.png")
    tbl_data = [
        [gaze_table, im2],
        [im3,        " "],
    ]
    style = [
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),        # 對齊
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),        # 對齊
        ('VALIGN', (0, 0), (1, -1), 'CENTER'),        # 對齊
        ('SPAN',(1,0),(1,-1))
        ]
    tbl = Table(tbl_data,colWidths = TABLE_WIDTH *inch/2, style=style)
    Element.append(tbl)
    return Element

def CUTReport(Element, CUT_Task):
    sub_head2 = sub_head("CUT Dynamic Eyeposition Tracking")
    sub_head3 = sub_head("Ocular Alignment Table")
    text1 = con_text("Cover Uncover Test Sequence in Primary Position")
    Quality_Bar = quality_bar(CUT_Task.OD, CUT_Task.OS)
    gaze_table = diagnose_table(CUT_Task)
    
    Element.append(sub_head2)
    im1 = EyeTrackImage(CUT_Task.saveImage_path+"\\DrawEyeTrack.png")
    Element.append(im1)
    Element.append(Quality_Bar)
    Element.append(sub_head3)
    Element.append(text1)    
    im2 = CutEyeImage(CUT_Task.saveImage_path+"\\DrawEyeFig.png")
    im3 = QRCodeImage(CUT_Task.saveImage_path+"\\QR_code.png")
    tbl_data = [
        [gaze_table, im2],
        [im3,        " "],
    ]
    style = [
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),        # 對齊
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),        # 對齊
        ('VALIGN', (0, 0), (1, -1), 'CENTER'),        # 對齊
        ('SPAN',(1,0),(1,-1))
        ]
    tbl = Table(tbl_data,colWidths = TABLE_WIDTH *inch/2, style=style)
    Element.append(tbl)
    return Element

def Gaze9Report(Element, Gaze9_Session):
    GAZE_9_STR      = nb.GAZE_9_TIME
    sub_head2 = sub_head("9 Gaze Dynamic Eyeposition Tracking")
    sub_head3 = sub_head("Ocular Motility Table")
    text1 = con_text("9 Gaze Test Sequence")
    Quality_Bar = quality_bar(Gaze9_Session.OD, Gaze9_Session.OS)
    
    im1 = EyeTrackImage(Gaze9_Session.saveImage_path+"\\DrawEyeTrack.png")
    im2 = Gaze9EyeImage(Gaze9_Session.saveImage_path+"\\DrawEyeFig.png")
    im3 = Gaze9EyeMesh(Gaze9_Session.saveImage_path+"\\DrawEyeMesh.png")
    im4 = QRCodeImage(Gaze9_Session.saveImage_path+"\\QR_code.png")
    Dev_H = Gaze9_Session.NeurobitDxDev_H
    Dev_V = Gaze9_Session.NeurobitDxDev_V
    Diff_H = Dev_H[:,0]-Dev_H[:,1]
    Diff_V = Dev_V[:,0]-Dev_V[:,1]
    data = [
        ["9 Gaze",  "OD ("+chr(176)+")",   "",        "OS ("+chr(176)+")",  "",     "OD-OS ("+chr(176)+")",    ""],
        ["",        "H",        "V",        "H",        "V",    "H",            "V"],
        [GAZE_9_STR[0], Dev_H[0][0], Dev_V[0][0], Dev_H[0][1], Dev_V[0][1], np.round(Diff_H[0],1), np.round(Diff_V[0],1)],
        [GAZE_9_STR[1], Dev_H[1][0], Dev_V[1][0], Dev_H[1][1], Dev_V[1][1], np.round(Diff_H[1],1), np.round(Diff_V[1],1)],
        [GAZE_9_STR[2], Dev_H[2][0], Dev_V[2][0], Dev_H[2][1], Dev_V[2][1], np.round(Diff_H[2],1), np.round(Diff_V[2],1)],
        [GAZE_9_STR[3], Dev_H[3][0], Dev_V[3][0], Dev_H[3][1], Dev_V[3][1], np.round(Diff_H[3],1), np.round(Diff_V[3],1)],
        [GAZE_9_STR[4], Dev_H[4][0], Dev_V[4][0], Dev_H[4][1], Dev_V[4][1], np.round(Diff_H[4],1), np.round(Diff_V[4],1)],
        [GAZE_9_STR[5], Dev_H[5][0], Dev_V[5][0], Dev_H[5][1], Dev_V[5][1], np.round(Diff_H[5],1), np.round(Diff_V[5],1)],
        [GAZE_9_STR[6], Dev_H[6][0], Dev_V[6][0], Dev_H[6][1], Dev_V[6][1], np.round(Diff_H[6],1), np.round(Diff_V[6],1)],
        [GAZE_9_STR[7], Dev_H[7][0], Dev_V[7][0], Dev_H[7][1], Dev_V[7][1], np.round(Diff_H[7],1), np.round(Diff_V[7],1)],
        [GAZE_9_STR[8], Dev_H[8][0], Dev_V[8][0], Dev_H[8][1], Dev_V[8][1], np.round(Diff_H[8],1), np.round(Diff_V[8],1)],
    ]
    dis_list = []
    for x in data:
        dis_list.append(x)
    style = [
        ('FONTNAME', (0, 0), (-1, -1), 'TSans'),    # 字體
        ('FONTSIZE', (0, 0), (-1, -1), 8),          # 字體大小
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),          # 字體顏色
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),          # 字體顏色
        
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),        # 對齊
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
        
        ('SPAN',(0,0),(0,1)),
        ('SPAN',(1,0),(2,0)),
        ('SPAN',(3,0),(4,0)),
        ('SPAN',(5,0),(6,0)),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),   # 設置表格框線爲grey色，線寬爲0.5
        
        ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
        ('BACKGROUND', (0, 0), (0, -1), colors.steelblue),
        ('BACKGROUND', (1, 1), (-1, 1), colors.powderblue)
    ]    
    gaze_table = Table(dis_list, style=style, rowHeights=12, colWidths=6/8.4 *inch) 
    tbl_data = [
        [im4, gaze_table]]
    tbl = Table(tbl_data)
    
    Element.append(sub_head2)    
    Element.append(im1)
    Element.append(Quality_Bar)
    Element.append(sub_head3)
    Element.append(text1)    
    Element.append(im2) 
    Element.append(im3)
    Element.append(tbl)
    return Element     

def Gaze9ACTReport(Element, Gaze9_Session):
    GAZE_9_STR      = nb.GAZE_9_TIME
    HEADER = ["9 Gaze Dynamic Eyeposition Tracking (Open)",
              "9 Gaze Dynamic Eyeposition Tracking (Cover Left)",
              "9 Gaze Dynamic Eyeposition Tracking (Cover Right)"]
    
    sub_head3 = sub_head("Ocular Motility Table")
    text1 = con_text("9 Gaze Test Sequence")
    Quality_Bar = quality_bar(Gaze9_Session.OD, Gaze9_Session.OS)
    
    
    im4 = QRCodeImage(Gaze9_Session.saveImage_path+"\\QR_code.png")
    Dev_H = Gaze9_Session.NeurobitDxDev_H
    Dev_V = Gaze9_Session.NeurobitDxDev_V
    
    for ss in range(0,3):
        data = [
            ["9 Gaze",  "OD ("+chr(176)+")",   "",        "OS ("+chr(176)+")",  "",     "OD-OS ("+chr(176)+")",    ""],
            ["",        "H",        "V",        "H",        "V",    "H",            "V"],
        ]
        sub_head2 = sub_head(HEADER[ss])
        im1 = EyeTrackImage(Gaze9_Session.saveImage_path+"\\DrawEyeTrack_"+nb.ACT_LABEL[ss]+".png")
        im2 = Gaze9EyeImage(Gaze9_Session.saveImage_path+"\\DrawEyeFig_"+nb.ACT_LABEL[ss]+".png")
        im3 = Gaze9EyeMesh(Gaze9_Session.saveImage_path+"\\DrawEyeMesh_"+nb.ACT_LABEL[ss]+".png")
        for t in GAZE_9_STR:
            Diff_H = Dev_H[t][ss][0]-Dev_H[t][ss][1]
            Diff_V = Dev_V[t][ss][0]-Dev_V[t][ss][1]
            data.append([t, Dev_H[t][ss][0], Dev_V[t][ss][0], Dev_H[t][ss][1], Dev_V[t][ss][1], np.round(Diff_H,1), np.round(Diff_V,1)])
        dis_list = []
        for x in data:
            dis_list.append(x)
        style = [
            ('FONTNAME', (0, 0), (-1, -1), 'TSans'),    # 字體
            ('FONTSIZE', (0, 0), (-1, -1), 8),          # 字體大小
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),          # 字體顏色
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),          # 字體顏色
            
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),        # 對齊
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),     # 對齊
            
            ('SPAN',(0,0),(0,1)),
            ('SPAN',(1,0),(2,0)),
            ('SPAN',(3,0),(4,0)),
            ('SPAN',(5,0),(6,0)),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),   # 設置表格框線爲grey色，線寬爲0.5
            
            ('BACKGROUND', (0, 0), (-1, 0), colors.steelblue),
            ('BACKGROUND', (0, 0), (0, -1), colors.steelblue),
            ('BACKGROUND', (1, 1), (-1, 1), colors.powderblue)
        ]    
        gaze_table = Table(dis_list, style=style, rowHeights=12, colWidths=6/8.4 *inch) 
        tbl_data = [
            [im4, gaze_table]]
        tbl = Table(tbl_data)
        
        Element.append(sub_head2)    
        Element.append(im1)
        Element.append(Quality_Bar)
        Element.append(sub_head3)
        Element.append(text1)    
        Element.append(im2) 
        Element.append(im3)
        Element.append(tbl)
        Element.append(PageBreak())
    return Element     
