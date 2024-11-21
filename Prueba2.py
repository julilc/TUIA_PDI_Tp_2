from rqst import *
import os

# Parte 1: Preparar lista de rutas para imágenes de patentes
PATENTES_PATH: list = [f"data/img0{i}.png" for i in range(1,10)]
for i in range(10,13):
    PATENTES_PATH.append(f"data/img{i}.png")
PATENTES_PATH

# Parte 2: Procesar imágenes para detectar posibles patentes
img_1 = cv2.imread(PATENTES_PATH[0])
imshow(img_1)

#Definimos ventana deslizante donde buscar conjunto de letras
w, h, m = img_1.shape
w = int(0.1)
h = int(0.1)
ventana = (w,h)

#Función para obtener estructura de imagen
img_gray = cv2.cvtColor(img_1.copy(), cv2.COLOR_BGR2GRAY)
_, img_1_bin = cv2.threshold(img_gray.copy(), 150, 255, type=cv2.THRESH_BINARY)
imshow(img_1_bin)

def encontrar_patente(img_bin, img):
    ext_contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = img.copy()
    cv2.drawContours(img_out, ext_contours, -1, (255,0,0), 2)
    height, width = img.shape[:2]
    posibles_patentes = []
    for c in ext_contours:
        # Aproximar el contorno a un polígono
        epsilon = 0.045 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        area = cv2.contourArea(c)
        # Verificar si el polígono tiene 4 vértices
        x, y, w, h = cv2.boundingRect(c)
        if len(approx) == 4 and 42 < w < 103 and 11 < h < 46:
            # Dibujar el contorno del rectángulo
            cv2.drawContours(img_out, [c], -1, (0, 255, 0), 2)
            # Ajustar las coordenadas del recorte
            y_start = max(0, y - int(h * 0.3))
            y_end = min(height, y + h + int(h * 0.2))
            patente = img[y_start:y_end, x:x+w]
            posibles_patentes.append(patente)
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    imshow(img_out)
    return posibles_patentes

posibles_patentes = []
for i in range(len(PATENTES_PATH)):
    img_ = cv2.imread(PATENTES_PATH[i])
    w = 4
    h = 9
    img_gray = cv2.cvtColor(img_.copy(), cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray.copy(), 63, 250, type=cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_bin, 200, 350)
        
    s = cv2.getStructuringElement(cv2.MORPH_RECT, (h,w))
    img_close = cv2.morphologyEx(img_canny.copy(), cv2.MORPH_CLOSE, s, iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_close, connectivity=8)
    filtered_img = np.zeros_like(img_close, dtype=np.uint8)
    for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= 1500:
            filtered_img[labels == i] = 255
    img_open = cv2.morphologyEx(filtered_img.copy(), cv2.MORPH_OPEN, s, iterations=3)
    posibles_patentes += encontrar_patente(img_open, img_)

print(len(posibles_patentes))
# Parte 3: Procesar una patente seleccionada para extraer caracteres
pats = [0,1,5,7,10,12,14,17,20,27,30]
for paty in pats:
    pat = posibles_patentes[paty]
    # Convertir la patente a escala de grises
    gris = cv2.cvtColor(pat, cv2.COLOR_BGR2GRAY)

    # Ecualizamos la imagen de manera local para resaltar mejor las letras
    m = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    eq_img = cv2.equalizeHist(gris)
    # imshow(eq_img)

    # Aplicamos Black Hat para diferenciar mejor el fondo
    img_black = cv2.morphologyEx(eq_img, cv2.MORPH_BLACKHAT, m, iterations=9)
    # imshow(img_black)

    # Binarizamos la imagen
    _, img_bin = cv2.threshold(img_black, 70, 255, cv2.THRESH_BINARY_INV)
    # imshow(img_bin)

    # Quitamos áreas muy grandes que corresponden al borde de la patente
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)
    filtered_img = np.zeros_like(img_bin, dtype=np.uint8)
    for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= 5 and stats[i, cv2.CC_STAT_AREA] <= 100:
            filtered_img[labels == i] = 255
    # imshow(filtered_img)

    # Aplicamos dilatación con un kernel vertical para cerrar las letras
    e = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    img_erode = cv2.morphologyEx(filtered_img, cv2.MORPH_DILATE, kernel=e, iterations=2)
    imshow(img_erode)

    # Dibujar componentes conectadas
    img_draw = pat.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_erode, connectivity=4)
    for i in range(1, num_labels):  # Comienza en 1 para omitir el fondo
        x, y, w, h, area = stats[i]  # Extraer información de la componente
        # Dibujar rectángulo y centroide
        color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 1)

    # Mostrar la imagen con las componentes conectadas dibujadas
    imshow(img_draw)