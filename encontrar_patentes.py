########################################################################
################### Consigna 1: detectar patentes ######################
########################################################################

### Path de trabajo
# 1: obtener estructura de imagen.
# 2: identificar parte que es letra.
# 3: detectar patente mediante ventana deslizante que
#   contenga 6 caracteres dentro.
# 4: recortar patente y letras.


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
w, h, m = img_1.shape
w = int(0.1)
h = int(0.1)
ventana = (w,h)

#Función para obtener estructura de imagen
img_gray = cv2.cvtColor(img_1.copy(),cv2.COLOR_BGR2GRAY)
_,img_1_bin = cv2.threshold(img_gray.copy(),150,255, type=cv2.THRESH_BINARY)
imshow(img_1_bin)

def encontrar_patente(img):
    ext_contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = cv2.merge((img,img,img))
    cv2.drawContours(img_out, ext_contours, -1, (255,0,0), 2)
    #imshow(img_out, title="Contorno exterior")
    posible_patente = []
    for c in ext_contours:
        # Aproximar el contorno a un polígono
        epsilon = 0.02 * cv2.arcLength(c, True)  # Ajusta 0.02 según lo necesario
        approx = cv2.approxPolyDP(c, epsilon, True)
        
        # Verificar si el polígono tiene 4 vértices
        if len(approx) == 4:
            # Dibujar el contorno del rectángulo
            cv2.drawContours(img_out, [c], -1, (0, 255, 0), 2)
            
            # Calcular el rectángulo delimitador y dibujarlo (opcional)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    imshow(img_out)

for i in range(len(PATENTES_PATH)):
    print(i)
    img_ = cv2.imread(PATENTES_PATH[i])
    img_gray = cv2.cvtColor(img_.copy(),cv2.COLOR_BGR2GRAY)
    _,img_1_bin = cv2.threshold(img_gray.copy(),150,255, type=cv2.THRESH_BINARY)
    imshow(img_1_bin)
    encontrar_patente(img_1_bin)
