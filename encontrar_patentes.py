from rqst import *
import os

def encontrar_rectangulos(img_bin, img):
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
    # imshow(img_out)
    return posibles_patentes


def encontrar_patente(posibles_pat : list[np.array])->bool:
    se_encontro_patente = False
    
    #Para cada rectángulo en la imagen obtenemos sus componentes
    for i in range(len(posibles_pat)):
        pat = posibles_pat[i]
        # Convertir la patente a escala de grises
        gris = cv2.cvtColor(pat, cv2.COLOR_BGR2GRAY)

        # Ecualizamos la imagen de manera local para resaltar mejor las letras
        m = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
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

        #Obtenemos las componentes conectadas y aplicamos filtro por área
        for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
            if stats[i, cv2.CC_STAT_AREA] >= 10 and stats[i, cv2.CC_STAT_AREA] <= 100:
                filtered_img[labels == i] = 255
               
        
        #imshow(filtered_img)

        # Aplicamos dilatación con un kernel vertical para cerrar las letras
        e = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
        img_dil = cv2.morphologyEx(filtered_img, cv2.MORPH_DILATE, kernel=e, iterations=1)

        #Utilizamos erosión para obtener una imagen con aquellas áreas que están uniendo las letras    
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        img_erode = cv2.morphologyEx(img_dil, cv2.MORPH_ERODE, kernel=kernel_horizontal, iterations=1)
        
        #Le quitamos a la imagen, las áreas que unen las letras
        filtered_img -=  img_erode
        #imshow(filtered_img)
        
        # Detectamos componentes conectadas
        img_draw = pat.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_img, connectivity=8)
        
        #Obtenemos la mediana de altura, posición y, y anchura
        hs = stats[1:, cv2.CC_STAT_HEIGHT] 
        median_altura = np.median(hs)
        ys = stats[1:, cv2.CC_STAT_TOP]
        median_y =np.median(ys)
        anchos = stats[1:, cv2.CC_STAT_WIDTH]  # Omitimos la etiqueta 0 (el fondo)
        mediana_anchos = np.median(anchos)
        
        #Como los caracteres de las patentes están en tamaños similares, aquellas componentes que estén 
        #cercanas a la mediana serán letras.
        
        caracteres = []

        for i in range(1, num_labels):  
            x, y, w, h, area = stats[i]

            #Chequeamos si esa componente es un caracter por su cercanía a las medianas.  
            if mediana_anchos-5< w < mediana_anchos+10 and median_altura - 7 < h < median_altura + 5 and median_y - 15 < y < median_y + 15:
                caracteres.append(pat[0:pat.shape[0], x:x+w])#Sumamos en 1 a la cantidad de caracteres
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 1)
            
        # Mostrar la imagen con los caracteres dibujados
        if len(caracteres) == 6:
            # Debe mostrar la imagen 'Pat' arriba y abajo en una fila los 6 caracteres que están en la lista 'caracteres'
            # Crear un espacio en blanco para separar los caracteres
            # Ordenar las componentes por la coordenada x (posición 'LEFT' del rectángulo)
            ordenadas = sorted(range(len(caracteres)), key=lambda i: stats[i + 1, cv2.CC_STAT_LEFT])

            # Crear una lista de caracteres ordenados
            caracteres = [caracteres[i] for i in ordenadas]

            altura_caracteres = caracteres[0].shape[0]
            espacio_img = np.ones((altura_caracteres, 4, 3), dtype=np.uint8) * 255  # Espacio en blanco con 3 canales

            # Concatenar caracteres con espacios
            fila_caracteres = []
            for i, char in enumerate(caracteres):
                fila_caracteres.append(char)
                if i < len(caracteres) - 1:  # No añadir espacio después del último carácter
                    fila_caracteres.append(espacio_img)

            # Concatenar horizontalmente los caracteres
            fila_caracteres = np.hstack(fila_caracteres)

            # Asegurarse de que la patente tenga el mismo ancho que la fila de caracteres
            ancho_fila = fila_caracteres.shape[1]
            pat_resized = cv2.resize(pat, (ancho_fila, pat.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Crear un espacio entre la patente y los caracteres (por ejemplo, 20 píxeles de altura)
            espacio_patente_caracteres = np.ones((3, pat_resized.shape[1], 3), dtype=np.uint8) * 255  # Espacio blanco de 20 píxeles

            # Combinar la patente con los caracteres
            imagen_final = np.vstack([pat_resized, espacio_patente_caracteres, fila_caracteres])  # Añadimos el espacio entre la patente y los caracteres

            # Mostrar la imagen combinada
            imshow(imagen_final, blocking=True)
            se_encontro_patente = True
    return se_encontro_patente
    

def detectar_patentes(img:np.array)-> np.array:

    posibles_patentes = []    
    w = 4
    h = 9
    
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(img_gray.copy(), 63, 250, type=cv2.THRESH_BINARY)
    img_canny = cv2.Canny(img_bin, 200, 350)
    
    #Aplicamos Close sobre Canny para unir la líneas cortadas    
    s = cv2.getStructuringElement(cv2.MORPH_RECT, (h,w))
    img_close = cv2.morphologyEx(img_canny.copy(), cv2.MORPH_CLOSE, s, iterations=2)
    
    #Detectamos las componentes conectadas y filtramos por aquellas con area chica
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_close, connectivity=8)
    filtered_img = np.zeros_like(img_close, dtype=np.uint8)
    for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= 1500:
            filtered_img[labels == i] = 255

    #Realizamos Open para dividir conexiones erróneas producto de Close.
    img_open = cv2.morphologyEx(filtered_img.copy(), cv2.MORPH_OPEN, s, iterations=3)
    
    #Lllamamos a función que detecta rectángulos
    posibles_patentes += encontrar_rectangulos(img_open, img)

    patente_encontrada = encontrar_patente(posibles_pat= posibles_patentes)
    
    if not patente_encontrada:
        print('Patente no encontrada')
       

def user()->np.array:
    path_img = input('ingrese la ubicación de su imagen: ')
    img = cv2.imread(path_img)
    detectar_patentes(img)

user()