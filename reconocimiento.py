import numpy as np  # Importa la biblioteca numpy para operaciones con matrices
from skimage.measure import label, regionprops  # Importa funciones para etiquetar regiones y obtener sus propiedades
import cv2  # Importa la biblioteca OpenCV para procesamiento de imágenes
import matplotlib.pyplot as plt  # Importa la biblioteca matplotlib para visualización
from skimage.morphology import skeletonize  # Importa la función skeletonize para obtener el esqueleto de una imagen

# Función para etiquetar componentes conectados en una imagen binaria
def etiquetar_componentes(imagen_binaria):
    print("Etiquetando componentes conectados...")
    # Etiqueta las regiones conectadas en la imagen binaria
    imagen_etiquetada = label(imagen_binaria, connectivity=2)
    # Obtiene el número de componentes etiquetados
    num_componentes = imagen_etiquetada.max()
    print(f"{num_componentes} componentes conectados etiquetados.")
    return imagen_etiquetada, num_componentes

# Función para mostrar dos imágenes lado a lado con Matplotlib
def mostrar_imagenes(imagen1, imagen2, titulo1, titulo2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Crea una figura con dos subgráficos
    axes[0].imshow(imagen1, cmap='gray')  # Muestra la primera imagen en escala de grises
    axes[0].set_title(titulo1)  # Establece el título del primer subgráfico
    axes[0].axis('off')  # Desactiva los ejes del primer subgráfico
    
    axes[1].imshow(imagen2, cmap='gray')  # Muestra la segunda imagen en escala de grises
    axes[1].set_title(titulo2)  # Establece el título del segundo subgráfico
    axes[1].axis('off')  # Desactiva los ejes del segundo subgráfico
    
    plt.show()  # Muestra la figura con las dos imágenes

# Función para encontrar el contorno más grande en una imagen binaria
def encontrar_contorno_mas_grande(imagen_binaria):
    # Asegura que la imagen esté en formato uint8 para OpenCV
    imagen_binaria = (imagen_binaria * 255).astype(np.uint8)
    # Encuentra todos los contornos en la imagen binaria
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Encuentra el contorno más grande basado en el área
    contorno_mas_grande = max(contornos, key=cv2.contourArea)
    return contorno_mas_grande

# Función para aplicar transformaciones (rotaciones y escalados) a una imagen
def aplicar_transformaciones(imagen):
    transformaciones = []  # Lista para almacenar las transformaciones

    # Escalados (0.8x, 1x, 1.2x)
    for escala in [0.8, 1, 1.2]:
        # Escala la imagen según el factor dado
        imagen_escalada = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST)
        
        # Rotaciones en ángulos de 0, 45, 90 grados
        for angulo in [0, 45, 90]:
            # Obtiene la matriz de rotación para el ángulo dado
            matriz_rotacion = cv2.getRotationMatrix2D((imagen_escalada.shape[1] // 2, imagen_escalada.shape[0] // 2), angulo, 1)
            # Aplica la rotación a la imagen escalada
            imagen_rotada = cv2.warpAffine(imagen_escalada, matriz_rotacion, (imagen_escalada.shape[1], imagen_escalada.shape[0]))
            # Añade la imagen rotada a la lista de transformaciones
            transformaciones.append(imagen_rotada)

    return transformaciones  # Devuelve la lista de imágenes transformadas

# Función para extraer y comparar características utilizando coincidencia de contornos
def reconocer_jeroglificos(imagen_etiquetada, referencias):
    print("Reconociendo jeroglíficos...")
    # Obtiene las propiedades de las regiones etiquetadas
    regiones = regionprops(imagen_etiquetada)
    jeroglificos_reconocidos = []  # Lista para almacenar los jeroglíficos reconocidos

    for i, region in enumerate(regiones):
        if region.area >= 50:  # Filtra pequeños componentes ruidosos
            print(f"\nProcesando región {i + 1}/{len(regiones)} con área {region.area}")
            # Obtiene la imagen de la región y aplica skeletonize para obtener el esqueleto
            region_imagen = region.image
            region_imagen = skeletonize(region_imagen).astype(np.uint8)

            # Aplica transformaciones a la región detectada
            transformaciones = aplicar_transformaciones(region_imagen)

            mejor_coincidencia = None
            menor_distancia = float('inf')

            for nombre, referencia in referencias.items():
                # Aplica skeletonize a la referencia
                referencia = skeletonize(referencia).astype(np.uint8)
                print(f"  Comparando con jeroglífico de referencia: {nombre}")

                for transformada in transformaciones:
                    # Encuentra el contorno más grande en la imagen transformada y la referencia
                    contorno_region = encontrar_contorno_mas_grande(transformada)
                    contorno_referencia = encontrar_contorno_mas_grande(referencia)
                    # Calcula la distancia de coincidencia de contornos
                    distancia = cv2.matchShapes(contorno_region, contorno_referencia, cv2.CONTOURS_MATCH_I1, 0.0)
                    #print(f"  Distancia con {nombre} (transformada): {distancia:.4f}")

                    # Muestra las imágenes comparadas
                    #mostrar_imagenes(transformada, referencia, f"Región Transformada - {nombre}", f"Referencia - {nombre}")

                    # Si la distancia es menor, actualiza la mejor coincidencia
                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        mejor_coincidencia = nombre

            # Añade la mejor coincidencia a la lista de reconocidos, incluso si no es una coincidencia fuerte
            jeroglificos_reconocidos.append(mejor_coincidencia)
            if menor_distancia >= 0.3:
                print(f"Jeroglífico {mejor_coincidencia} reconocido con una distancia de {menor_distancia:.4f}, pero la coincidencia no es fuerte.")
            else:
                print(f"Jeroglífico {mejor_coincidencia} reconocido con una distancia de {menor_distancia:.4f}.")

    return jeroglificos_reconocidos  # Devuelve la lista de jeroglíficos reconocidos
