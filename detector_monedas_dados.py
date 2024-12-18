########################################################################
### Consigna 1: detectar monedas y dados con sus respectivos valores ###
########################################################################

#Parte 0: carga de librerías.
from rqst import *

#### Parte 1: detectar círculos y cuadrados
### (o monedas y dados en el contexto del problema)


### Funcion a Utilizar para ajustar imagen original.
def adjust_hsv_image_with_lightness_and_threshold(img: np.array, scale: float = 0.5):
    """
    Ajusta los valores HSVL de una imagen en tiempo real utilizando OpenCV.
    También devuelve los valores finales de H, S, V, L y Threshold.

    Retorna:
    - modified_bgr (numpy.ndarray): Imagen con ajustes de HSVL.
    - thresh_image (numpy.ndarray): Imagen binarizada con el umbral ajustado.
    - h, s, v, l, t (int): Valores finales de Hue, Saturation, Value, Lightness y Threshold.
    """
    # Redimensionar la imagen original según el factor de escala
    resized_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.blur(hsv_img, (15, 15))  # Aplicamos un suavizado para evitar ruidos

    # Crear ventanas
    cv2.namedWindow("Modified Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Thresholded Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Adjustments", cv2.WINDOW_NORMAL)

    # Contenedor para almacenar los valores de los sliders
    slider_values = {'h': 0, 's': 0, 'v': 0, 'l': 0, 't': 0}

    # Variables para almacenar las imágenes
    modified_bgr = None
    thresh_image = None

    # Función de actualización para aplicar cambios
    def update_image(x):
        nonlocal modified_bgr, thresh_image

        # Actualizar valores de los sliders
        slider_values['h'] = cv2.getTrackbarPos('Hue', 'Adjustments')
        slider_values['s'] = cv2.getTrackbarPos('Saturation', 'Adjustments')
        slider_values['v'] = cv2.getTrackbarPos('Value', 'Adjustments')
        slider_values['l'] = cv2.getTrackbarPos('Lightness', 'Adjustments')
        slider_values['t'] = cv2.getTrackbarPos('Threshold', 'Adjustments')

        # Aplicar ajustes a la copia de la imagen
        modified_hsv = hsv_img.copy()
        modified_hsv[:, :, 0] = np.clip(hsv_img[:, :, 0] + slider_values['h'], 0, 179)
        modified_hsv[:, :, 1] = np.clip(cv2.blur(hsv_img[:, :, 1] + slider_values['s'], (5, 5)), 0, 255)
        modified_hsv[:, :, 2] = np.clip(hsv_img[:, :, 2] + slider_values['v'], 0, 255)
        modified_hsv[:, :, 2] = np.clip(cv2.blur(modified_hsv[:, :, 2] + slider_values['l'], (5, 5)), 0, 255)

        # Convertir la imagen modificada de HSV a BGR
        modified_bgr = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)

        # Convertir a escala de grises y aplicar umbral
        gray_image = cv2.cvtColor(modified_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh_image = cv2.threshold(gray_image, slider_values['t'], 255, cv2.THRESH_BINARY)

        # Mostrar las imágenes
        cv2.imshow('Modified Image', modified_bgr)
        cv2.imshow('Thresholded Image', thresh_image)

    # Crear sliders
    cv2.createTrackbar('Hue', 'Adjustments', 50, 100, update_image)
    cv2.createTrackbar('Saturation', 'Adjustments', 50, 100, update_image)
    cv2.createTrackbar('Value', 'Adjustments', 50, 100, update_image)
    cv2.createTrackbar('Lightness', 'Adjustments', 50, 100, update_image)
    cv2.createTrackbar('Threshold', 'Adjustments', 127, 255, update_image)

    # Inicializar ventana
    update_image(0)

    # Mantener la ventana abierta hasta que se presione ESC
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Código ASCII para ESC
            break

    cv2.destroyAllWindows()

    # Retornar la imagen modificada, la binarizada y los valores finales
    return modified_bgr, thresh_image, slider_values['h'], slider_values['s'], slider_values['v'], slider_values['l'], slider_values['t']

##Encontramos figuras ###

##Función para encontrar círculos
def find_and_draw_circles(thresh_image, dp_ = 1.2, minD = 30, p1 = 50, p2 = 100, minR = 50, maxR = 500)-> tuple[np.array, list]:
    """
    #### Función para encontrar círculos en una imagen binarizada (thresh_image)
    #### y dibujarlos sobre una nueva imagen.
    ----------------------------------------------------------------------
    #### Parámetros:
        - thresh_image (numpy.ndarray): Imagen binarizada en la que se buscarán los círculos.
        - dp _ : valor de dp para HoughCircles.
        - minD: distancia mínima entre un centroide de un círculo y otro.
        - p1: parámetro 1 de HoughCircles.
        - p2: parámetro 2 de HoughCircles a mayor número, más exigente en la detección de círculos.
        - minR: radio mínimo de círculo.
        - maxR: radio máximo de círculo.

    ----------------------------------------------------------------------

    #### Retorna:
        - result_image (numpy.ndarray): Imagen con los círculos dibujados sobre la original.
        - circulos: circulos encontrados: (list[(x,y,r)])
    
    ----------------------------------------------------------------------

    #### Procedimiento:
        1. Encuentra los círculos en la imagen mediante HoughCircles.
        2. Dibuja los círculos en la imagen.
        3. Retorna la imagen con los círculos dibujados y una lista de dichos
            círculos.
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
            area = np.pi * r **2
            print(area)
            # Dibujar el círculo exterior
            cv2.circle(result_image, (x, y), r, (0, 255, 0), 4)
            # Dibujar el centro del círculo
            cv2.circle(result_image, (x, y), 2, (0, 0, 255), 3)

            # Escribir el área del círculo en la imagen
            cv2.putText(result_image, f'Area: {area:.2f}', (x - 40, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
        
        # Mostrar la imagen con los círculos dibujados
        #imshow(result_image)
    
    else:
        print("No se encontraron círculos en la imagen.")

    return result_image, circles



### Si bien el approach anterior sirve para detectar las monedas, no así los dados
### por lo tanto hacemos una función que detecte ambos

def encontrar_cuadrados_y_circulos(img: np.array, min_fp : int = 0.06, max_fp : int = 0.08)-> tuple[list, list]:
    '''
    Esta función encuentra cuadrados y círculos en una imagen, utilizando el factor
    de forma para diferenciar los segundos de los primeros.
    --------------------------------------------------------
    
    ### Parámetros:
        - img: imagen donde detectar las figuras.
        - min_fp: tolerancia mínima de factor de forma para círculo.
        - max_fp: tolerancia máxima de factor de forma para círculo.

    ---------------------------------------------------------
    ### Retorna:
        - Lista cuadrados.
        - Lista círculos con sus áreas.

    ---------------------------------------------------------
    ### Procedimiento:
        1. Obtiene la binarización de la imagen.
        2. Detecta bordes con Canny.
        3. Encuentra los contornos.
        4. Mediante el factor de forma del círculo clasifica los contornos,
            dibuja en distintos colores cada figura diferenciada junto con su área
            y arma las respectivas listas de circulos y cuadrados.
    '''
    _, th = cv2.threshold(img.copy(), 50,190, cv2.THRESH_BINARY)
    # Detectar bordes con Canny
    edges = cv2.Canny(img, 50, 150)

    #imshow(edges)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    draw_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cuadrados = []
    monedas = []
    # Procesar cada contorno
    for contour in contours:
        # Aproximar el contorno para simplificarlo
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        p = cv2.arcLength(contour, True)
        fp = area / (p**2)

        if min_fp<fp<max_fp:
            shape = "Círculo"
            color = (0,255,0)
            area  *=  np.pi
            monedas.append((contour, area))
        else:  
            cuadrados.append(contour)
            shape = "cuadrado"
            color = (255,0,0)

        x,y = contour[0,0]
        
        # Dibujar el contorno y la etiqueta
        cv2.drawContours(draw_img, [contour], -1, color, 2)
            # Escribir el área del círculo en la imagen
        cv2.putText(draw_img, f'{shape}Area: {area:.2f}', (x-40, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    #imshow(draw_img)
    return cuadrados, monedas


## Parte 2: clasificar monedas.

def clasificar_monedas(monedas: list[tuple]) -> dict:
    '''
    #### Esta función clasifica los tipos de monedas.

    -------------------------------------------------
    #### Parámetros:
        - monedas: lista de monedas, cada una representada 
            como una tupla (contorno, area de la moneda).
    
    -------------------------------------------------
    
    #### Retorna:
        - dic_monedas: diccionario {moneda: cantidad}
        - total_dinero: total de dinero por la suma de las monedas.
    
    -------------------------------------------------

    #### Procedimiento:
        1. A cada moneda se la clasifica según su área en 10 cent, 50 cent
        y 1 peso.
        2. Según su clasificación se suma en 1 al diccionario la moneda.
        3. Según su clasificación se suma el monto en dinero que dicha moneda
            representa.


    '''
    area_1_peso = 65500
    area_10_cent = 41550
    area_50_cent = 80500
    dict_monedas = {'1 peso': 0, '50 cent': 0, '10 cent': 0}
    total_dinero = 0
    for m in monedas:
        area = m[1]
        if area_1_peso-3500<area<area_1_peso+3500:
            dict_monedas['1 peso'] += 1
            total_dinero += 1
        if area_50_cent-1500 <area<area_50_cent+1500:
            dict_monedas['50 cent'] += 1
            total_dinero += 0.5
        if area_10_cent-5550 < area < area_10_cent+5550:
            dict_monedas['10 cent'] +=1
            total_dinero += 0.1
    
    total_dinero = round(total_dinero,2)

    return dict_monedas, total_dinero



## Parte 3: contar puntos de dados.

def contar_dados(dados, thr_img) -> tuple[dict, int]:
    '''
    Esta función devuelve para cada dado su puntaje y
    el puntaje total obtenido por todos los dados.
    -------------------------------------------------------

    #### Parámetros:
        - dados: lista de dados, cada uno representado como un contorno.

    -------------------------------------------------------

    #### Retorna:
        -dict_dados: diccionrio {dado: puntaje}
        -puntaje: total de puntos obtenidos entre todos los dados.        
    
    -------------------------------------------------------

    #### Procedimiento:
        1. Para cada dado se busca sus círculos (es decir sus puntos)
            utilizando HoughCircles.
        2. Se cuenta la cantidad de puntos y se los suma en el diccionario al dado i.
        3. Se suma al total de puntos cada punto detectado en dicho dado.
    '''

    dict_dados = {}
    puntaje = 0
    i = 1
    for dado in dados:
        x, y, w, h = cv2.boundingRect(dado)
        area = cv2.contourArea(dado)
        if area < 500:
            continue
        d = thr_img[y:y+h, x:x+w]
        #imshow(d)

        # Detectar círculos en la región recortada
        circles = cv2.HoughCircles(d, cv2.HOUGH_GRADIENT, dp=1.3, minDist=10,
                                param1=10, param2=22, minRadius=1, maxRadius=30)
        
        punto_dado = 0
        img_draw = cv2.cvtColor(d.copy(), cv2.COLOR_GRAY2BGR)

        # Asegurarse de que se encontraron círculos
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")  # Redondear y convertir a enteros
            # Dibujar los círculos encontrados
            for (x, y, r) in circles:
                area = np.pi * r ** 2
                # Dibujar el círculo exterior
                cv2.circle(img_draw, (x, y), r, (0, 255, 0), 1)
                # Dibujar el centro del círculo
                cv2.circle(img_draw, (x, y), 2, (0, 0, 255), 1)
        
                punto_dado += 1
        #imshow(img_draw)
        dict_dados[f"dado {i}"] = punto_dado
        puntaje += punto_dado
        i += 1
    return dict_dados, puntaje

def user():


    path_img = input('ingrese la dirección donde se encuentra guardada la imagen:')

    img: np.array = cv2.imread(path_img)  # Carga la imagen en BGR
    
    ######Aplica las transformaciones en los valores necesarias##########
    # Convertir la imagen de BGR a HSV

    hsv_img: np.array = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    resized_img = cv2.resize(hsv_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.blur(hsv_img, (15, 15))

    
    
    ### Ajustes manuales según los valores óptimos obtenidos
    modified_hsv = hsv_img.copy()

    # Ajuste del valor del Hue, asegurándose de que esté dentro de su rango [0, 179]
    modified_hsv[:, :, 0] = np.clip(hsv_img[:, :, 0] + 26, 0, 179)  # Hue

    # Ajuste de Saturation, asegurándose de que esté dentro del rango [0, 255]
    modified_hsv[:, :, 1] = np.clip(cv2.blur(hsv_img[:, :, 1] + 0, (5, 5)), 0, 255)  # Saturation

    # Ajuste de Value, asegurándose de que esté dentro del rango [0, 255]
    modified_hsv[:, :, 2] = np.clip(hsv_img[:, :, 2] + 0, 0, 255)  # Value

    # Aplicar ajuste de Lightness sobre el canal Value (V), asegurándose de no exceder los límites
    modified_hsv[:, :, 2] = np.clip(cv2.blur(modified_hsv[:, :, 2] + 0, (5, 5)), 0, 255)  # Lightness

    # Convertir la imagen modificada de HSV a BGR
    modified_bgr = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)
    # Convertir la imagen modificada a escala de grises
    gray_image = cv2.cvtColor(modified_bgr, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización (Threshold) para obtener una imagen binaria con el valor ajustado
    _, thresh_image = cv2.threshold(gray_image, 63, 255, cv2.THRESH_BINARY)

    #imshow(thresh_image)
    ######Encontrar formas ##########
    ## Se observa que la imagen presenta "ruido", definido este como aquellas zonas 
    ## que no son ni dados ni monedas.

    ## Realizamos un filtrado según el área para descartar la mayor parte de este ruido.
    ## De no hacer esto, luego al aplicar morfología dicho ruido podría unir figuras, cosa
    ## que complicaría la posterior clasificación.

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_image, connectivity=8)

    # Crea una máscara para conservar solo los componentes con área mayor o igual a 1300 píxeles
    filtered_img = np.zeros_like(thresh_image, dtype=np.uint8)
    for i in range(1, num_labels):  # Omite el fondo (etiqueta 0)
        if stats[i, cv2.CC_STAT_AREA] >= 1300:
            filtered_img[labels == i] = 255  # Mantiene el componente en la imagen filtrada


    #imshow(filtered_img)

    ##Se obtiene una imagen con menor ruido donde se puede ahora aplicar morfología.

    ##Primero aplicamos close para cerrar los dados, es decir que los puntos de los mismos
    ## queden de color blanco; y además también para completar mejor algunas monedas.

    s: np.array = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    close_img: np.array =cv2.morphologyEx(filtered_img.copy(), cv2.MORPH_CLOSE, s, iterations=9)
    #imshow(close_img)

    ## Luego realizamos Open para dividir figuras que se juntaron debido a la operación anterior.

    open_img: np.array = cv2.morphologyEx(close_img.copy(), cv2.MORPH_OPEN, s, iterations=9)

    #imshow(open_img)
    ## Como se puede ver, se obtiene una imagen con sus figuras rellenas y separadas.

    ##Llamamos a la función para encontrar las monedas y dados pasándole esta última imagen.

    dados, monedas = encontrar_cuadrados_y_circulos(open_img, min_fp = 0.063, max_fp = 0.075)

    ### De aquí obtenemos también el área de cada moneda aproximadamente
    ## moneda 1 peso: 62000, , 66100, , 68785, 68844
    ## moneda 10 centavos: 45506, 44929, 36924, 45169, 46366,45983, 47094,
    ##                      44395, 44998
    ## Moneda 50 centavos: 80663, 79740, 81846

    ##Las de 1 peso tienen areas desde 62000 a 69000 aprox, siendo su punto medio 65500
    ## Las de 10 centavos tienen areas desde 36000 a 47100 aprox, siendo su punto medio 41500
    ### Las de 50 centavos, van desde 79000 a 82000, siendo su punto medio 80500

    ################### CLASIFICAR MONEDAS ##########################################
    
    dic_monedas , total_dinero = clasificar_monedas(monedas)

    ################### CONTAR DADOS ################################################
    dict_dados, puntaje = contar_dados(dados, thresh_image)
    
    ########## RESULTADOS ##############################
    print('Las monedas encontradas fueron')
    for key, value in dic_monedas.items():
        print(f'Tipo de moneda: {key}, total encontradas: {value}')
    print(f'El monto total es de ${total_dinero}')
    print('Los puntos por dado fueron')
    for key, value in dict_dados.items():
        print(f'{key}, total puntos: {value}')
    print(f'El puntaje total es {puntaje}')


user()


