print("\nSE ESTÁN CARGANDO LAS FUNCIONES......................................................")

import functions as f
import os

print("\nLAS FUNCIONES SE HAN CARGADO CORRECTAMENTE, SE PROCEDE CON LA EJECUCIÓN DEL MAIN....\n")

print("\nSe guardan las ground truths de las imágenes que se van a emplear.")

# Guadamos las anotaciones de las imágenes que se encuentren en la carpeta original_images
anotations = f.save_annotations_original_images()

#Inicialmente calculamos el mAP del dataset de imágenes sin transformar
num_img_transformed = 1 
transformation = [[0]] #al elegir 0 no aplicamos ninguna transformación
transformed_images = f.transformations(num_img_transformed,transformation)
#simplificamos las boxes con el Non-maximum Suppression
f.apply_NMS()   
#Ejecutamos el proyecto de las métricas para obtener el mAP

print("\n##############################################################################")
print("mAP IMÁGENES SIN TRANSFORMAR:")
os.system('python -W ignore "./Object-Detection-Metrics-master/pascalvoc.py" ')
print("##############################################################################\n")

#Ejecutamos una transformación simple. El ajuste empleado sera de brillo.
num_img_transformed = 1
#elegimos la transformación, en este caso transformación de brillo
transformation = [[2]]                                                      
#obtenemos el dataframe de las detecciones del dataset transformado
transformed_images = f.transformations(num_img_transformed,transformation)    
#simplificamos las boxes con el Non-maximum Suppression
f.apply_NMS()          
#calculamos el mAP ejecutando el proyecto de las métricas     
print("\n##############################################################################")
print("mAP utilizando una transformación simple de brillo:")                                               
os.system('python -W ignore "./Object-Detection-Metrics-master/pascalvoc.py" ')
print("##############################################################################\n")

#Ejecutamos una transformacion compuesta sobre cada imágen. Los ajustes utilizados serán de brillo + contraste.
num_img_transformed = 1
#elegimos una transformación de brillo + contraste
transformation = [[2,3]] 
#obtenemos el dataframe de las detecciones del dataset transformado
transformed_images = f.transformations(num_img_transformed,transformation)
#simplificamos las boxes con el Non-maximum Suppression
f.apply_NMS()
f.time.sleep(2)
#calculamos el mAP ejecutando el proyecto de las métricas 
print("\n##############################################################################")
print("mAP utilizando una transformación compuesta de brillo + contraste:")
os.system('python -W ignore "./Object-Detection-Metrics-master/pascalvoc.py" ')
print("##############################################################################\n")

#Ejecutamos varias transformaciones simples y las unimos con NMS. Los ajustes empleados serán de saturación y color.
num_img_transformed = 1
#elegimos una transformación de saturación y una transformación de contraste
transformation = [[4],[5]] 
#obtenemos el dataframe de las detecciones del dataset con las detecciones de cada una de las transformaciones simples
transformed_images = f.transformations(num_img_transformed,transformation)
#simplificamos las boxes con el Non-maximum Suppression
f.apply_NMS()
f.time.sleep(2)
#calculamos el mAP ejecutando el proyecto de las métricas
print("\n##############################################################################")
print("mAP utilizando dos transformaciones simples, una de saturación y otra de color:")
os.system('python -W ignore "./Object-Detection-Metrics-master/pascalvoc.py" ')
print("##############################################################################\n")
print("Final del programa.............................................................")