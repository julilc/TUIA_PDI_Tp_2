########################################################################
################### Consigna 1: detectar patentes ######################
########################################################################

#Parte 0: carga de librerías e imágenes.
from rqst import *
import os

PATENTES_PATH: list = [f"data/img0{i}.png" for i in range(1,10)]
for i in range(10,13):
    PATENTES_PATH.append(f"data/img{i}.png")
PATENTES_PATH

img_1 = cv2.imread(PATENTES_PATH[0])
imshow(img_1)

# Parte 2: capturar letras.

#Definimos ventana deslizante donde buscar conjunto de letras
w, h = img_1.shape
w = int(0.1)
h = int(0.1)
ventana = (w,h)

#Función para obtener estructura de imagen
def obtener_estructura(img: np.array)-> np.array:
    