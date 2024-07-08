import numpy as np
from skimage.measure import label, regionprops
import cv2
import matplotlib.pyplot as plt

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

# Función para extraer y comparar características utilizando similitud de formas
def reconocer_jeroglificos(imagen_etiquetada, referencias):
    print("Reconociendo jeroglíficos...")
    regiones = regionprops(imagen_etiquetada)
    jeroglificos_reconocidos = []
    
    for region in regiones:
        if region.area >= 50:  # Filtrar pequeños componentes ruidosos
            region_imagen = region.image
            mejor_coincidencia = None
            mayor_similitud = 0

            for nombre, referencia in referencias.items():
                # Redimensionar la región al tamaño de la referencia manteniendo las proporciones
                region_redimensionada = cv2.resize(region_imagen.astype(np.uint8), (referencia.shape[1], referencia.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Mostrar las imágenes de la región redimensionada y la referencia
                mostrar_imagenes(region_redimensionada, referencia, f"Región Redimensionada - {nombre}", f"Referencia - {nombre}")

                # Calcular la similitud (utilizamos correlación cruzada)
                similitud = np.sum(region_redimensionada == referencia) / referencia.size
                print(f"Similitud con {nombre}: {similitud:.2f}")

                if similitud > mayor_similitud:
                    mayor_similitud = similitud
                    mejor_coincidencia = nombre

            if mayor_similitud > 0.7:  # Umbral de similitud
                jeroglificos_reconocidos.append(mejor_coincidencia)
                print(f"Jeroglífico {mejor_coincidencia} reconocido con una similitud de {mayor_similitud:.2f}.")
            else:
                jeroglificos_reconocidos.append("?")
                print("Jeroglífico no reconocido.")

    return jeroglificos_reconocidos
