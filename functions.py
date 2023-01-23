#Imports
import random as rd
import numpy as np
from PIL import Image, ImageEnhance
import torch
import pandas as pd
import tensorflow as tf
import json
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
import os
import math
import time

#Funciones para transformar las imágenes

#Brillo
def add_brightness(img,num_img,flag,model):
    enhacer = ImageEnhance.Brightness(img)
    factors = [1]
    lista = []
    while 1 in factors:
        factors = np.random.uniform(0.5,1.75,num_img)
    for fact in factors:
        lista.append(np.asarray(enhacer.enhance(fact)))
    if flag == 0:
        return get_all_detections(lista,model)
    else:
        return lista

#Contraste
def add_contrast(img,num_img,flag,model):
    enhacer = ImageEnhance.Contrast(img)
    factors = [1]
    lista = []  
    while 1 in factors:
        factors = np.random.uniform(0.5,2,num_img)
    for fact in factors:
        lista.append(np.asarray(enhacer.enhance(fact)))
    if flag == 0:
        return get_all_detections(lista,model)
    else:
        return lista

#Saturación
def add_sharpness(img,num_img,flag,model):
    enhacer = ImageEnhance.Sharpness(img)
    factors = np.random.uniform(0.5,5,num_img)
    lista = []
    for fact in factors:
        lista.append(np.asarray(enhacer.enhance(fact)))
    if flag == 0:
        return get_all_detections(lista,model)
    else:
        return lista

#Color
def add_color(img,num_img,flag,model):
    enhacer = ImageEnhance.Color(img)
    factors = [1]
    lista = []
    while 1 in factors:
        factors = np.random.uniform(0,2.5,num_img)
    for fact in factors:
        lista.append(np.asarray(enhacer.enhance(fact)))
    if flag == 0:
        return get_all_detections(lista,model)
    else:
        return lista

#Rotaciones
def genera_rotaciones(img,num_rot,flag,model):
    dict_rotations = dict()
    rotaciones = []
    grados = [355,356,357,358,359,1,2,3,4,5] #+-10º
#     grados = [357,358,359,1,2,3,87,88,89,90,91,92,93,177,178,179,180,181,182,183,267,268,269,270,271,272,273]
#     grados = [350,351,352,353,354,355,356,357,358,359,1,2,3,4,5,6,7,8,9,10] #+-15º
    for i in range(0,num_rot):
        rand_idx = rd.randint(0, len(grados)-1)
        rot = rotate_image(np.asanyarray(img),grados[rand_idx])
        shp = np.asarray(img)
        dict_rotations[i] = (grados[rand_idx],shp.shape[:2])
        rotaciones.append(rot)
        #write_params(grados[rand_idx])
    df_aux = get_all_detections(rotaciones,model)
    return deshacer_rot(df_aux,dict_rotations)
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Funciones auxiliares para rotar y deshacer las rotaciones de las imágenes
def rotate_box(corners,angle,  cx, cy, h, w): 
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0])))) 
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    calculated = np.dot(M,corners.T).T
    calculated = calculated.reshape(-1,8)
    return calculated

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width/2, height/2) 
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def deshacer_rot(df_rotado,dict_angle):
    df_final = pd.DataFrame()
    for i in range(0,len(df_rotado)): #Recorremos el numero de detecciones distintas--
        df_aux = df_rotado[df_rotado['Deteccion']==i]
        for j in range(0,len(df_aux)):
            angle = dict_angle[i][0]
            y,x = dict_angle[i][1]
            a = get_enclosing_box(rotate_box(get_corners(np.asarray([[df_aux.loc[j,'xmin'],
                                                                      df_aux.loc[j,'ymin'],
                                                                      df_aux.loc[j,'xmax'],
                                                                      df_aux.loc[j,'ymax']]])),
                                             -angle,y/2,x/2,x,y))
            df_aux.loc[j,'xmin'] = limit(a[0][0],x)
            df_aux.loc[j,'ymin'] = limit(a[0][1],y)
            df_aux.loc[j,'xmax'] = limit(a[0][2],x)
            df_aux.loc[j,'ymax'] = limit(a[0][3],y)
        df_final = pd.concat([df_aux,df_final],axis=0)
    return df_final

def get_corners(bboxes):
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    x2 = x1 + width
    y2 = y1 
    x3 = x1
    y3 = y1 + height
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    return corners

def get_enclosing_box(corners):
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    return final

def limit(pos,lim):
    if pos>lim:
        return lim
    elif pos < 0:
        return 0
    return pos
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Funciones para generar el dataset transformado
def transformations(num,elec):
    remove_detections()
    time.sleep(5)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    df_transformed = pd.DataFrame()
    for img_path in os.listdir('./original_images'):
        img = Image.open('original_images/'+img_path)
        #write_params("\n")
        for trans in elec:
            if len(trans) == 1: #solo se realiza una transformación
                df_aux = elige_funcion(img,num,trans[0],0,model)
                write_transformations(df_aux,img_path)
                df_transformed = pd.concat([df_aux,df_transformed],axis=0)
            else: #se realizan varias transformaciones
                list_img = []
                list_img_aux = []
                rot = False
                for idx,elem in enumerate(trans):
                    if idx == 0:
                        list_img_aux = elige_funcion(img,num,elem,1,model)
                    elif elem == 1: #salimos para llamar a las rotaciones fuera
                        rot = True
                        break
                    else:
                        list_img_aux = list_img
                        list_img = []
                    for i in range(0,len(list_img_aux)):
                        list_img.extend(elige_funcion(Image.fromarray(list_img_aux[i]),1,elem,1,model)) 
                if rot is True:
                    for img_aux in list_img: #caso de las rotaciones
                        df_aux = elige_funcion(Image.fromarray(img_aux),1,1,0,model)
                        write_transformations(df_aux,img_path)
                        df_transformed = pd.concat([df_aux,df_transformed],axis=0)
                else:
                    df_aux = get_all_detections(list_img,model)
                    write_transformations(df_aux,img_path)
                    df_transformed = pd.concat([df_aux,df_transformed],axis=0)
    return df_transformed

def elige_funcion(img,num,transform,flag,model):
    if transform == 0:
        return get_all_detections([np.asarray(img)],model)
    if transform == 1:
        return genera_rotaciones(img,num,flag,model)
    elif transform == 2:
        return add_brightness(img,num,flag,model)
    elif transform == 3:
        return add_contrast(img,num,flag,model)
    elif transform == 4:
        return add_sharpness(img,num,flag,model)
    elif transform == 5:
        return add_color(img,num,flag,model)
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Funciones para escribir en la carpeta de las métricas
def write_transformations(df,image_path):
    image_id = re.search(r"2014_0*(\d*)",image_path)
    image_id = image_id.group(1)
    for i in range(0,len(df)):
        f = open("./Object-Detection-Metrics-master/detections/"+str(image_id)+".txt", "a")
        f.write(df.iloc[i]['name']+" "+str(df.iloc[i]['confidence'])+" "+str(df.iloc[i]['xmin'])+" "+str(df.iloc[i]['ymin'])+" "+
                str(df.iloc[i]['xmax']-df.iloc[i]['xmin'])+" "+str(df.iloc[i]['ymax']-df.iloc[i]['ymin']))
        f.write("\n")
        f.close()

def write_exec(transformation,num_img_transformed,mAp):
    f = open("./output.txt", "a")
    f.write(str(transformation)+"; num_img:"+
            str(num_img_transformed)+"; num_orig:"+
            str(len(os.listdir('./original_images')))+"; "+
            mAp)
    f.write("\n")
    f.close()
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Funciones para obtener las anotaciones de cada imágen y escribirlas en la carpeta de las métricas
def save_annotations_original_images():
    lista_original = os.listdir('./original_images')
    lista_org_final = []
    for elem in lista_original:
        m = re.match(r"([^0-9]*2014_0*)([0-9]*)",elem)
        lista_org_final.append(m.group(2))

    lista_gt = os.listdir('./Object-Detection-Metrics-master/groundtruths/')
    lista_gt = [x[0:-4] for x in lista_gt]
    for elem in lista_gt:
        if not elem in lista_org_final:
            os.remove('./Object-Detection-Metrics-master/groundtruths/'+elem+'.txt')
            
    all_boxes = []
    for path in os.listdir('./original_images'):
        img = Image.open('original_images/'+path)
        all_boxes.append(get_annotations(img))
    return all_boxes   

def get_annotations(image_id):
    
    all_classes = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
                   9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'street sign', 13: 'stop sign', 14: 'parking meter',
                   15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant',
                   23: 'bear', 24: 'zebra', 25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe',
                   30: 'eye glasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
                   37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
                   43: 'tennis racket', 44: 'bottle', 45: 'plate', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
                   50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
                   57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
                   64: 'potted plant', 65: 'bed', 66: 'mirror', 67: 'dining table', 68: 'window', 69: 'desk',
                   70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
                   83: 'blender', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
                   90: 'toothbrush', 91: 'hair brush'}
    
    yolo_classes = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
                    'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13,
                    'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21,
                    'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28,
                    'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34,
                    'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39,
                    'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46,
                    'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53,
                    'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60,
                    'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67,
                    'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74,
                    'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
    
    if type(image_id)!=int:
        m = re.match(r"([^0-9]*2014_0*)([0-9]*)",image_id.filename)
        image_id = int(m.group(2))
    try:
        with open("./Object-Detection-Metrics-master/groundtruths/"+str(image_id)+'.txt','r') as t:
            return t.read().split('\n')[0:-1]
    except:
        f = open('./annotations/instances_val2014.json')
        dat = json.load(f) #procesamos el archivo JSON que tiene todas las detecciones
        boxes = [] #Creamos una lista porque ese archivo tendra para cada imagen las detecciones por separado
        category = []
        class_name = []
        for i in dat['annotations']:
            if i['image_id'] == image_id:
                category.append(yolo_classes[all_classes[i['category_id']]])
                class_name.append(all_classes[i['category_id']])
                boxes.append(i['bbox'])
        f.close()
        f = open("./Object-Detection-Metrics-master/groundtruths/"+str(image_id)+".txt", "w")
        for i,box in enumerate(boxes):
            f.write(class_name[i])
            for elem in box:
                f.write(" ")
                f.write(str(elem))
            f.write("\n")
        f.close()
        boxes.append(category)
        return boxes
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Función que proporciona como entrada al modelo un dataset y devuelve las detecciones realizadas
def get_all_detections(dataset,model):
    df = pd.DataFrame()
    for i in range(0,len(dataset)):
        imgs = Image.fromarray(dataset[i])
        results = model(imgs)
        aux = pd.DataFrame(results.pandas().xyxy[0])
        aux = aux.assign(Deteccion=i)        
        df = pd.concat([aux,df],axis=0)  
    return df 
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Funciones que aplican el NMS para simplificar las detecciones en las transformaciones compuestas con varias imágenes
def apply_NMS():
    folder_path = './Object-Detection-Metrics-master/detections/'
    for path in os.listdir(folder_path):
        df_final = pd.DataFrame()
        f = open(folder_path+path, "r")
        df = str_to_df(f.read().split("\n"))
        f.close()
        boxes = get_box(df)
        for box in boxes:
            df_final = pd.concat([df_final,df.iloc[box[0][0]].to_frame().T],ignore_index=True)
        df = df_final
        f = open(folder_path+path,"w")
        for i in range(0,len(df_final)):
            f.write(df.iloc[i]['name']+" "+str(df.iloc[i]['confidence'])+" "+str(df.iloc[i]['xmin'])+" "+str(df.iloc[i]['ymin'])+" "+
                    str(df.iloc[i]['xmax']-df.iloc[i]['xmin'])+" "+str(df.iloc[i]['ymax']-df.iloc[i]['ymin']))
            f.write("\n")
        f.close()

def str_to_df(st):
    df = pd.DataFrame(columns =['xmin','ymin','xmax','ymax','name','confidence','class']) 
    yolo_classes = {'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9, 'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 'toothbrush': 79}
    for det in st[:-1]:
        det = det.split(' ')
        if re.match(r"[a-zA-Z]+",det[1]):
            new_row = {'xmin':[float(det[3])],'ymin':[float(det[4])],'xmax':[float(det[3])+float(det[5])],'ymax':[float(det[4])+float(det[6])],'name':[det[0]+" "+det[1]],'confidence':[float(det[2])],'class':[yolo_classes[det[0]+" "+det[1]]]}
        else:
            new_row = {'xmin':[float(det[2])],'ymin':[float(det[3])],'xmax':[float(det[2])+float(det[4])],'ymax':[float(det[3])+float(det[5])],'name':[det[0]],'confidence':[float(det[1])],'class':[yolo_classes[det[0]]]}
        df_aux = pd.DataFrame(new_row)
        df = pd.concat([df,df_aux],ignore_index=True,axis=0)
    return df

def get_box(aux):
    boxes = []
    for name in pd.unique(list(aux['name'])): #recorremos cada una de las detecciones encontradas
        df_aux = aux[(aux['name']==name) & (aux['confidence']>0.3)]  ########## CONFIDENCE ###########
        df_aux = np.array([list(df_aux['xmin']),list(df_aux['ymin']),list(df_aux['xmax']),list(df_aux['ymax']),list(df_aux['confidence'])])
        df_aux = df_aux.transpose()
        if len(df_aux>0):
            if boxes is None:
                boxes = [NMS(df_aux),str(aux[aux['name']==name]['class'].unique()[0])]
            else:
                boxes.append([NMS(df_aux),str(aux[aux['name']==name]['class'].unique()[0])])
    return boxes
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Algoritmo Non-maximum suppression
def NMS(dets, thresh=0.75):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Función para dibujar las bounding boxes sobre una imagen
def draw_bounding_boxes(image,boxes):
    if type(boxes[0]) == np.ndarray:
        boxes = list(boxes)
    colors = [(165,0,42),(205,173,0),(0,255,255),(178,58,238),(0,0,0),(0,0,139),(142,229,238),(127,255,0),(240,248,255)]
    color = 0
    for box_aux in boxes:
        box = []
        for i in box_aux:
            box.append(int(i))
        image = cv2.rectangle(copy.deepcopy(image),box[:2],box[2:], colors[color], 2)
        color +=1
        
    image = Image.fromarray(image)
    return image

def draw_from_df(img_path,df):
    img = np.asanyarray(Image.open(img_path))
    boxes = []
    for i in range(0,len(df)):
        box = []
        box.append(df.iloc[i][0])
        box.append(df.iloc[i][1])
        box.append(df.iloc[i][2])
        box.append(df.iloc[i][3])
        boxes.append(box)
    image = draw_bounding_boxes(img,boxes)
    return image
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Función para limpiar la carpeta de las métricas para cada ejecución
def remove_detections():
    dir = './Object-Detection-Metrics-master/detections/'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Función que escribe los parámetros con los que se ha ejecutado cada transformación
def write_params(params):
    if not type(params) == str:
        params = str(params)
    f = open("./out.txt", "a")
    f.write(params)
    f.write(" ")
    f.close()