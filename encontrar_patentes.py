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

def encontrar_patente(img_bin, img):
    ext_contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = img.copy()
    cv2.drawContours(img_out, ext_contours, -1, (255,0,0), 2)
    height, width = img.shape[:2]
    #imshow(img_out, title="Contorno exterior")
    posibles_patentes = []
    for c in ext_contours:
        # Aproximar el contorno a un polígono
        epsilon = 0.045 * cv2.arcLength(c, True)  # Ajusta 0.02 según lo necesario
        approx = cv2.approxPolyDP(c, epsilon, True)
        area = cv2.contourArea(c)
        # Verificar si el polígono tiene 4 vértices
        x, y, w, h = cv2.boundingRect(c)
        if len(approx) == 4 and 42 < w < 103 and 11<h<46:
            # Dibujar el contorno del rectángulo
            cv2.drawContours(img_out, [c], -1, (0, 255, 0), 2)
            # Ajustar las coordenadas del recorte
            y_start = max(0, y - int(h * 0.3))  # Margen superior (máximo 0)
            y_end = min(height, y + h + int(h * 0.2))  # Margen inferior (máximo altura total)
            patente = img[y_start:y_end, x:x+w]
            imshow(patente)
            # Calcular el rectángulo delimitador y dibujarlo (opcional)
            
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #imshow(img_out)
    return posibles_patentes

for i in range(len(PATENTES_PATH)):
    print(i)
    img_ = cv2.imread(PATENTES_PATH[i])
    w = 4
    h = 9
    print(h,w)
    img_gray = cv2.cvtColor(img_.copy(),cv2.COLOR_BGR2GRAY)
    _,img_bin = cv2.threshold(img_gray.copy(),63,250, type=cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_bin, 200,350)
    #imshow(img_canny)
    #imshow(img_canny)
        
    s = cv2.getStructuringElement(cv2.MORPH_RECT, (h,w))

    img_close = cv2.morphologyEx(img_canny.copy(), cv2.MORPH_CLOSE, s, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_close, connectivity=8)
    filtered_img = np.zeros_like(img_close, dtype=np.uint8)
    for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= 1500:
            filtered_img[labels == i] = 255
    img_open = cv2.morphologyEx(filtered_img.copy(), cv2.MORPH_OPEN, s, iterations= 3)
    #imshow(img_close)
    #imshow(img_open)
    posibles_patentes = encontrar_patente(img_open, img_)
print(posibles_patentes)
### anchos: 44, 65,73 , 81, 74, 65, 102, 74,74, 67, 75 (min =44, max = 102)
### altos:  13, 21, 22, 28, 35, 28, 45, 28, 33, 29, 28 (min = 13, max = 45)


def filtrar_patentes_con_area(posibles_patentes):
    # Lista para almacenar las patentes válidas
    patentes_validas = []
    
    for patente_img in posibles_patentes:
        img_gray = cv2.cvtColor(patente_img, cv2.COLOR_BGR2GRAY)
        
        _, img_bin = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        # valid_contours = []
        # for c in contours:
        #     area = cv2.contourArea(c)
        #     if area > 1 and area < 10:  
        #         valid_contours.append(c)
        
        if contours > 6 and len(contours) < 10 :
            patentes_validas.append(patente_img)
        
            plt.imshow(patente_img)
            plt.show()

    return patentes_validas
filtrar_patentes_con_area(posibles_patentes)