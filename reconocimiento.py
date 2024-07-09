import numpy as np
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# Función para etiquetar componentes conectados
def etiquetar_componentes(imagen_binaria):
    print("Etiquetando componentes conectados...")
    imagen_etiquetada = label(imagen_binaria, connectivity=2)
    num_componentes = imagen_etiquetada.max()
    print(f"{num_componentes} componentes conectados etiquetados.")
    return imagen_etiquetada, num_componentes

# Función para mostrar dos imágenes lado a lado con Matplotlib
def mostrar_imagenes(imagen1, imagen2, titulo1, titulo2):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(imagen1, cmap='gray')
    axes[0].set_title(titulo1)
    axes[0].axis('off')
    
    axes[1].imshow(imagen2, cmap='gray')
    axes[1].set_title(titulo2)
    axes[1].axis('off')
    
    plt.show()

# Función para encontrar el contorno más grande en una imagen binaria
def encontrar_contorno_mas_grande(imagen_binaria):
    # Asegurarse de que la imagen esté en formato uint8 para OpenCV
    imagen_binaria = (imagen_binaria * 255).astype(np.uint8)
    contornos, _ = cv2.findContours(imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno_mas_grande = max(contornos, key=cv2.contourArea)
    return contorno_mas_grande

# Función para aplicar transformaciones (rotaciones y escalados) a una imagen
def aplicar_transformaciones(imagen):
    transformaciones = []

    # Escalados (0.8x, 1x, 1.2x)
    for escala in [0.8, 1, 1.2]:
        imagen_escalada = cv2.resize(imagen, None, fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST)
        
        # Rotaciones en ángulos de 0, 45, 90 grados
        for angulo in [0, 45, 90]:
            matriz_rotacion = cv2.getRotationMatrix2D((imagen_escalada.shape[1] // 2, imagen_escalada.shape[0] // 2), angulo, 1)
            imagen_rotada = cv2.warpAffine(imagen_escalada, matriz_rotacion, (imagen_escalada.shape[1], imagen_escalada.shape[0]))
            transformaciones.append(imagen_rotada)

    return transformaciones

# Función para extraer y comparar características utilizando coincidencia de contornos
def reconocer_jeroglificos(imagen_etiquetada, referencias):
    print("Reconociendo jeroglíficos...")
    regiones = regionprops(imagen_etiquetada)
    jeroglificos_reconocidos = []

    for i, region in enumerate(regiones):
        if region.area >= 50:  # Filtrar pequeños componentes ruidosos
            print(f"\nProcesando región {i + 1}/{len(regiones)} con área {region.area}")
            region_imagen = region.image
            region_imagen = skeletonize(region_imagen).astype(np.uint8)

            # Aplicar transformaciones a la región detectada
            transformaciones = aplicar_transformaciones(region_imagen)

            mejor_coincidencia = None
            menor_distancia = float('inf')

            for nombre, referencia in referencias.items():
                referencia = skeletonize(referencia).astype(np.uint8)
                print(f"  Comparando con jeroglífico de referencia: {nombre}")

                for transformada in transformaciones:
                    contorno_region = encontrar_contorno_mas_grande(transformada)
                    contorno_referencia = encontrar_contorno_mas_grande(referencia)
                    distancia = cv2.matchShapes(contorno_region, contorno_referencia, cv2.CONTOURS_MATCH_I1, 0.0)
                    #print(f"  Distancia con {nombre} (transformada): {distancia:.4f}")

                    #mostrar_imagenes(transformada, referencia, f"Región Transformada - {nombre}", f"Referencia - {nombre}")

                    if distancia < menor_distancia:
                        menor_distancia = distancia
                        mejor_coincidencia = nombre

            jeroglificos_reconocidos.append(mejor_coincidencia)
            if menor_distancia >= 0.3:
                print(f"Jeroglífico {mejor_coincidencia} reconocido con una distancia de {menor_distancia:.4f}, pero la coincidencia no es fuerte.")
            else:
                print(f"Jeroglífico {mejor_coincidencia} reconocido con una distancia de {menor_distancia:.4f}.")

    return jeroglificos_reconocidos
