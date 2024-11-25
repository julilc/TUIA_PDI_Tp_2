from rqst import *


def encontrar_rectangulos(img_bin: np.array, img:np.array)-> list[np.array]:
    '''
    Esta función recibe una imagen binaria y detecta en ella rectángulos que
    cumplan con especificaciones de ancho y alto, de forma tal que sean
    rectángulos horizontales similares a patentes.
    -------------------------------------------------------------------------
    ### Parámetros:
        - img_bin: imagen binaria donde buscar rectángulos.
        - img: imagen original de la cual se recortará el rectángulo obtenido.
    
    --------------------------------------------------------------------------

    ### Devuelve: 
        - posibles_patentes [list]: lista que contiene rectángulos recortados de
        la imagen original que pueden ser posibles patentes.

    ---------------------------------------------------------------------------

    ## Procedimiento:
        1. Extrae los contornos de la imagen binaria.
        2. Aproxima los rectángulos en la imagen.
        3. Filtra los rectángulos por ancho, alto (42 < w < 103 and 11 < h < 46)
        y cantidad de vértices (4)
        4. Los recorta de la imagen original y los agrega a la lista 
        "posibles_patentes".
        5. Retorna la lista "posibles_patentes".
    '''
    ext_contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_out = img.copy()
    #img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_out, ext_contours, -1, (255,0,0), 2)
    height, width = img.shape[:2]
    #imshow(img_out, title= 'Contornos extraidos en funcion encontrar rect.')
    posibles_patentes = []

    for c in ext_contours:

        # Aproximar el contorno a un polígono
        epsilon = 0.045 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)

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
    #imshow(img_out, title='5. Rectángulos encontrados')

    return posibles_patentes


def encontrar_patente(posibles_pat : list[np.array], img_original: np.array)->bool:
    '''
    Esta función recibe una lista de imágenes que son consideradas "posibles patentes"
    de allí detecta aquellas que lo son y recorta sus caracteres.

    ----------------------------------------------------------------------------------
    ### Parámetros:
        - posibles_pat [list[np.array]]: lista con imágenes que son consideradas
        posibles patentes.
        - img_original: imagen original para fines meramente ilustrativos.
    
    ----------------------------------------------------------------------------------

    ## Retorna:
        -bool : retorna True o False dependiendo de si se encontró o no una patente
        en la lista de posibles patentes.

    -----------------------------------------------------------------------------------

    ## Procedimiento:
        Para cada posible patente:
        1. Se la pasas a escala de grises.
        2. Se le aplica ecualización local.
        3. Se realiza blackhat con kernel (40,40) para diferenciar el fondo de las letras
        de forma más clara.
        4. Se obtiene imagen binaria.
        5. Se filtra aquellas areas pequeñas de la imagen obteniendo una imagen filtrada.
        6. Se detectan componentes conectadas y, si la imagen posee 6 componentes conectadas
        similares tanto en anchura, altura como en posición y:
            i. Se muestra la imagen original con el título "imagen recibida".
            ii. Se muestra una imagen con la patente recortada y los 6 caracteres debajo con
            el título "Patente y caracteres encontrados".
        7. Retorna False si no se encontró patente en la lista y True en caso de que sí. 


    '''
    
    se_encontro_patente = False
    
    #Para cada rectángulo en la imagen obtenemos sus componentes
    for i in range(len(posibles_pat)):
        pat = posibles_pat[i]

        gris = cv2.cvtColor(pat, cv2.COLOR_BGR2GRAY)

        m = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
        eq_img = cv2.equalizeHist(gris)
        #imshow(eq_img, title='1. Imagen ecualizada localmente')

        # Aplicamos Black Hat para diferenciar mejor el fondo
        img_black = cv2.morphologyEx(eq_img, cv2.MORPH_BLACKHAT, m, iterations=9)
        #imshow(img_black, title='2. Imagen con blackhat aplicado')

        _, img_bin = cv2.threshold(img_black, 70, 255, cv2.THRESH_BINARY_INV)
        #imshow(img_bin, title='3. Imagen blackhat binarizada')
        
        # Quitamos áreas muy grandes que corresponden al borde de la patente
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)
        
        filtered_img = np.zeros_like(img_bin, dtype=np.uint8)

        #Obtenemos las componentes conectadas y aplicamos filtro por área
        for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
            if stats[i, cv2.CC_STAT_AREA] >= 10 and stats[i, cv2.CC_STAT_AREA] <= 100:
                filtered_img[labels == i] = 255
        #imshow(filtered_img, title= '4. Imagen Filtrada')


        #imshow(filtered_img, title='5. Imagen filtrada - imagen erosionada')
        
        # Detectamos componentes conectadas
        img_draw = pat.copy()
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filtered_img, connectivity=8)
        
        #Obtenemos la mediana de altura, posición y, y anchura
        hs = stats[1:, cv2.CC_STAT_HEIGHT] 
        median_altura = np.median(hs)
        ys = stats[1:, cv2.CC_STAT_TOP]
        median_y = np.median(ys)
        anchos = stats[1:, cv2.CC_STAT_WIDTH]  # Omitimos la etiqueta 0 (el fondo)
        mediana_anchos = np.median(anchos)
        
        #Como los caracteres de las patentes están en tamaños similares, aquellas componentes que estén 
        #cercanas a la mediana serán letras.
        
        caracteres = []
        caracteres_coordenadas = []
        for i in range(1, num_labels):  
            x, y, w, h, area = stats[i]
            if w < 4 or h<6:
                continue
            
            #Chequeamos si esa componente es un caracter por su cercanía a las medianas.  
            if mediana_anchos-3< w < mediana_anchos+10 and median_altura - 7 < h < median_altura + 5 and median_y - 15 < y < median_y + 15:
                caracteres.append(pat[0:pat.shape[0], x:x+w])
                caracteres_coordenadas.append((x, pat[0:pat.shape[0], x:x+w]))
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 1)
        
        #imshow(img_draw, title='6. Caracteres encontrados en la imagen')
        
        # Mostrar la imagen con los caracteres dibujados
        if len(caracteres) == 6:
            # Debe mostrar la imagen 'Pat' arriba y abajo en una fila los 6 caracteres que están en la lista 'caracteres'
            # Crear un espacio en blanco para separar los caracteres
            # Ordenar las componentes por la coordenada x (posición 'LEFT' del rectángulo)
            caracteres_ordenados = sorted(caracteres_coordenadas, key=lambda char: char[0])
            caracteres = [char[1] for char in caracteres_ordenados]
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
            imshow(img_original, title='7. Imagen recibida', blocking= True)
            imshow(imagen_final, blocking=True, title='8. Patente y caracteres detectados')
            se_encontro_patente = True
    return se_encontro_patente
    

def detectar_patentes(img:np.array)-> None:
    '''
    Esta función recibe una imagen, le aplica transformaciones morfológicas y
    ejecuta las funciones 'encontrar_rectangulos' y 'encontrar_patente';
    en caso de que esta ultima retorne False, imprime 'patente no encontrada'

    --------------------------------------------------------------------------
    ### Parámetros:
        - img: imagen en formato rgb.
    
    --------------------------------------------------------------------------
    ### Retorna:
        None
    
    --------------------------------------------------------------------------
    ### Procedimiento:
        1. define ancho y alto del kernel.
        2. Toma la imagen y la pasa a escala de grises.
        3. Binariza la imagen.
        4. Obtiene el Canny.
        5. Aplica clausura sobre la imagen Canny para unir lineas.
        6. Obtiene componentes conectadas y filtra la imagen con clausura para
        eliminar componentes con áreas muy pequeñas.
        7. aplica apertura sobre la imagen filtrada para dividir uniones no
        deseadas.
        8. Llama a la función encontrar_rectangulos y le pasa como argumento la
        imagen filtrada y la original.
        9. Con las posibles patentes encontradas en la llamada anterior, ejecuto
        ejecuta la función patente_encontrada pasándolas como argumento.
        10. Si de la llamada anterior recibe como retorno un False, imprime 'Patente
        no encontrada'
    
    
    '''
    img_bgr = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    posibles_patentes = []    
    w, h= 4, 9
    
    img_gray = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2GRAY)
    #imshow(img_gray, title='Imagen en escala de grises')
    _, img_bin = cv2.threshold(img_gray.copy(), 63, 250, type=cv2.THRESH_BINARY)
    #imshow(img_bin, title='Imagen en escala de grises binarizada')
    img_canny = cv2.Canny(img_bin, 200, 350)
    #imshow(img_canny, title='1. Imagen Canny')
    
    #Aplicamos Close sobre Canny para unir la líneas cortadas    
    s = cv2.getStructuringElement(cv2.MORPH_RECT, (h,w))
    img_close = cv2.morphologyEx(img_canny.copy(), cv2.MORPH_CLOSE, s, iterations=2)
    #imshow(img_close, title='2. Imagen Canny con clausura')

    #Detectamos las componentes conectadas y filtramos por aquellas con area chica
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_close, connectivity=8)
    filtered_img = np.zeros_like(img_close, dtype=np.uint8)
    for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= 1500:
            filtered_img[labels == i] = 255

    #imshow(filtered_img, title='3. Imagen con clausura y filtrado')

    #Realizamos Open para dividir conexiones erróneas producto de Close.
    img_open = cv2.morphologyEx(filtered_img.copy(), cv2.MORPH_OPEN, s, iterations=3)
    #imshow(img_open, title='4. Imagen filtrada con apertura')
    #Lllamamos a función que detecta rectángulos
    posibles_patentes += encontrar_rectangulos(img_open, img)

    patente_encontrada = encontrar_patente(posibles_pat= posibles_patentes, img_original = img)
    
    if not patente_encontrada:
        print('Patente no encontrada')
       

def user()->np.array:
    '''
    Esta funcion py 

    --------------------------------------------------------------------------
    ### Parámetros:
        None

    --------------------------------------------------------------------------
    ### Retorna:
        None
    
    --------------------------------------------------------------------------
    ### Procedimiento:
        1. Le solicita la ubicación del archivo al usuario.
        2. Carga la imagen en BGR.
        3. La pasa como argumento a la función detectar_patentes.

    '''
    path_img = input('ingrese la ubicación de su imagen: ')
    img = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
    detectar_patentes(img)
user()
