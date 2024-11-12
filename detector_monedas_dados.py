########################################################################
### Consigna 1: detectar monedas y dados con sus respectivos valores ###
########################################################################
#### PROBAR CON ####
# ######## Puede obtenerse una imagen
# con un background uniforme
# substrayendo la apertura de la
# imagen original. Esta operación
# se denomina Transformación
# top-hat

###HSL

from PIL import Image 
#Parte 0: carga de librerías e imágenes.
from rqst import *
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
img_1 = cv2.imread('data/monedas.jpg')
img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2HSV)
#imshow(img_1, color_img= True, colorbar= 'HSV')

import cv2
import numpy as np

import cv2
import numpy as np

def adjust_hsv_image_with_lightness_and_threshold(img, scale=0.5):
    """
    Función para ajustar los valores HSVL de una imagen en tiempo real utilizando OpenCV.
    También muestra la versión binarizada de la imagen modificada en una ventana separada,
    con la posibilidad de ajustar el umbral dinámicamente.
    
    Parámetros:
    img (numpy.ndarray): Imagen en formato BGR (como la cargada con cv2.imread).
    scale (float): Factor de escala para redimensionar la imagen y ajustar la ventana.
    
    Retorna:
    modified_bgr (numpy.ndarray): Imagen con ajustes de HSVL.
    thresh_image (numpy.ndarray): Imagen binarizada con el umbral ajustado.
    """
    # Redimensionar la imagen original según el factor de escala
    resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    
    hsv_img = cv2.blur(hsv_img, (15, 15))  # Aplicamos un suavizado para evitar ruidos

    # Crear ventana para la imagen modificada y su versión binarizada
    cv2.namedWindow("Modified Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Thresholded Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Adjustments", cv2.WINDOW_NORMAL)  # Crear la ventana de ajustes

    # Redimensionar las ventanas
    cv2.resizeWindow("Modified Image", int(resized_img.shape[1] * 2), int(resized_img.shape[0]))
    cv2.resizeWindow("Thresholded Image", int(resized_img.shape[1] * 2), int(resized_img.shape[0]))
    cv2.resizeWindow("Adjustments", 400, 200)  # Tamaño para la ventana de ajustes

    # Variables para almacenar las imágenes modificadas y binarizadas
    modified_bgr = None
    thresh_image = None

    # Función de actualización para aplicar cambios
    def update_image(x):
        nonlocal modified_bgr, thresh_image  # Usamos las variables definidas fuera de la función

        # Obtener valores de H, S, V, L y Threshold de los deslizadores
        h = cv2.getTrackbarPos('Hue', 'Adjustments')
        s = cv2.getTrackbarPos('Saturation', 'Adjustments') 
        v = cv2.getTrackbarPos('Value', 'Adjustments') 
        l = cv2.getTrackbarPos('Lightness', 'Adjustments')# Ajuste para Lightness
        threshold_value = cv2.getTrackbarPos('Threshold', 'Adjustments')  # Valor del umbral

        # Aplicar ajustes a la copia de la imagen
        modified_hsv = hsv_img.copy()

        # Ajuste del valor del Hue, asegurándose de que esté dentro de su rango [0, 179]
        modified_hsv[:, :, 0] = np.clip(hsv_img[:, :, 0] + h, 0, 179)  # Hue

        # Ajuste de Saturation, asegurándose de que esté dentro del rango [0, 255]
        modified_hsv[:, :, 1] = np.clip(cv2.blur(hsv_img[:, :, 1] + s , (5,5)), 0, 255)  # Saturation

        # Ajuste de Value, asegurándose de que esté dentro del rango [0, 255]
        modified_hsv[:, :, 2] = np.clip(hsv_img[:, :, 2] + v, 0, 255)  # Value

        # Aplicar ajuste de Lightness sobre el canal Value (V), asegurándose de no exceder los límites
        modified_hsv[:, :, 2] = np.clip(cv2.blur(modified_hsv[:, :, 2] + l, (5,5)), 0, 255)  # Modificar solo el valor (V)

        # Convertir la imagen modificada de HSV a BGR
        modified_bgr = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

        # Convertir la imagen modificada a escala de grises
        gray_image = cv2.cvtColor(modified_bgr, cv2.COLOR_BGR2GRAY)

        # Aplicar umbralización (Threshold) para obtener una imagen binaria con el valor ajustado
        _, thresh_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Mostrar la imagen modificada en la ventana "Modified Image"
        cv2.imshow('Modified Image', modified_bgr)

        # Mostrar la imagen binarizada en la ventana "Thresholded Image"
        cv2.imshow('Thresholded Image', thresh_image)

    # Crear deslizadores para Hue, Saturation, Value, Lightness y Threshold con valores iniciales
    cv2.createTrackbar('Hue', 'Adjustments', 50, 100, update_image)
    cv2.createTrackbar('Saturation', 'Adjustments', 50, 100, update_image)
    cv2.createTrackbar('Value', 'Adjustments', 50, 100, update_image)
    cv2.createTrackbar('Lightness', 'Adjustments', 50, 100, update_image)  # Slider para Lightness
    cv2.createTrackbar('Threshold', 'Adjustments', 127, 255, update_image)  # Slider para Threshold

    # Llamar a la función de actualización para mostrar la imagen original al inicio
    update_image(0)

    # Mantener la ventana abierta hasta que se presione ESC
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Código ASCII para ESC
            break

    cv2.destroyAllWindows()

    # Al presionar ESC, devolver las imágenes modificadas
    return modified_bgr, thresh_image

mod_bgr , thr_img = adjust_hsv_image_with_lightness_and_threshold(img_1)

#### Llegamos a que los valore más óptimos son:
### hue 40, sat 18, v 9, l = 53 /**64/ 70, tr = 43
#### Obtenemos imagenes con dichos valores

### Dilatar y erosionar

imshow(thr_img)

#### Aplicamos primero una erosion para quitar ruido, es decir, aquello
### que no forma parte ni de una moneda ni de un dado

s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
img_close = cv2.morphologyEx(thr_img.copy(), cv2.MORPH_ERODE, s, iterations=15)
img_dil = cv2.morphologyEx(img_close.copy(), cv2.MORPH_CLOSE, s, iterations=1)
imshow(img_dil)
## Aplicamos un blur que servirá a la hora de detectar círculos.
blur_th = cv2.blur(img_dil.copy(), (25,25))
imshow(blur_th)


##Encontramos figuras ###

##Función para encontrar círculos
def find_and_draw_circles(thresh_image, dp_ = 1.2, minD = 30, p1 = 50, p2 = 100, minR = 50, maxR = 500):
    """
    Función para encontrar círculos en una imagen binarizada (thresh_image)
    y dibujarlos sobre una nueva imagen.
    
    Parámetros:
    thresh_image (numpy.ndarray): Imagen binarizada (escala de grises) en la que se buscarán los círculos.
    
    Retorna:
    result_image (numpy.ndarray): Imagen con los círculos dibujados sobre la original.
    """
    # Aplicar la Transformada de Hough para detectar círculos
    # El primer parámetro es la imagen de entrada, debe ser en escala de grises o binarizada
    circles = cv2.HoughCircles(thresh_image, 
                               cv2.HOUGH_GRADIENT, dp=dp_, minDist=minD, 
                               param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
    
    # Si se encuentran círculos
    if circles is not None:
        # Convertir las coordenadas de los círculos a enteros
        circles = np.round(circles[0, :]).astype("int")
        
        # Crear una imagen copia de la original para dibujar los círculos
        result_image = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para poder dibujar círculos
        
        # Dibujar los círculos encontrados
        for (x, y, r) in circles:
            # Dibujar el círculo exterior
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 4)
            # Dibujar el centro del círculo
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)
        
        # Mostrar la imagen con los círculos dibujados
        imshow(result_image)
    
    else:
        print("No se encontraron círculos en la imagen.")

    return result_image, circles

# Ejemplo de uso:
# Asumiendo que ya tienes la imagen binarizada 'thresh_image' de la función anterior:
result_image, circles = find_and_draw_circles(blur_th,p1 = 20, p2 = 50, minR=60, minD=170, maxR=100)

## Función para encontrar Cuadrados

img_canny = cv2.Canny(thr_img.copy(), 1, 1)

# Muestra la imagen con bordes detectados
imshow(img_canny)

C = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
canny_close = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, C, iterations=4)
imshow(canny_close)
# O = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# canny_open = cv2.morphologyEx(canny_close, cv2.MORPH_OPEN, O)
# imshow(canny_open)



# Detectar contornos
contornos, _ = cv2.findContours(canny_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_draw = cv2.cvtColor(canny_close.copy(), cv2.COLOR_GRAY2BGR)  # Convertir a BGR para dibujar en color
cv2.drawContours(img_draw.copy(),contornos, -1, color = (255,0,0),thickness=3)
imshow(img_draw)