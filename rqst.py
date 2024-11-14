#Librerías
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from PIL import Image 

def binarize(img: np.array, tr: int = 150, maxv: int = 255) -> np.array:
    '''
    Esta función recibe una imagen y la binariza según vlaores de tr y maxv recibidos.
    img: imagen en escala de grises
    tr : thresh.
    maxv: máximo valor de imagen de salida.
    '''
    _, img_bin = cv2.threshold(img, tr, maxv, cv2.THRESH_BINARY_INV)
    return img_bin



# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    #plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)