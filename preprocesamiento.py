import os
import cv2
import numpy as np

# Definir jeroglíficos de referencia
jeroglificos_referencia = {
    "ankh": "A",
    "wedjat": "J",
    "djed": "D",
    "scarab": "S",
    "was": "W",
    "akeht": "K"
}

# Cargar jeroglíficos de referencia
def cargar_jeroglificos_referencia():
    referencias = {}
    for nombre in jeroglificos_referencia.keys():
        ruta_imagen = f'jeroglificos/binarized/{nombre}.png'
        print(f"Intentando cargar {ruta_imagen}...")
        if not os.path.exists(ruta_imagen):
            print(f"Advertencia: No se encuentra el archivo de referencia: {ruta_imagen}")
            continue
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is None:
            print(f"Error: No se pudo cargar la imagen de referencia: {ruta_imagen}")
            raise FileNotFoundError(f"No se pudo cargar la imagen de referencia: {ruta_imagen}")
        _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY_INV)
        referencias[nombre] = binaria
        print(f"Imagen de referencia {nombre} cargada correctamente.")
    if not referencias:
        raise RuntimeError("No se pudieron cargar las imágenes de referencia.")
    return referencias

# Función para preprocesar la imagen
def preprocesar_imagen(ruta_imagen):
    print(f"Intentando cargar la imagen de entrada: {ruta_imagen}")
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encuentra el archivo de entrada: {ruta_imagen}")
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen de entrada: {ruta_imagen}")
    _, binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY_INV)
    print("Imagen de entrada cargada y binarizada correctamente.")
    return binaria
