import matplotlib.pyplot as plt
from skimage.color import label2rgb

# Función para mostrar la imagen binarizada
def mostrar_imagen_binaria(imagen_binaria, titulo):
    plt.figure(figsize=(6, 6))
    plt.imshow(imagen_binaria, cmap='gray')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Función para mostrar los objetos detectados coloreados
def mostrar_etiquetas_coloreadas(imagen_etiquetada, imagen, titulo):
    imagen_coloreada = label2rgb(imagen_etiquetada, image=imagen, bg_label=0)
    plt.figure(figsize=(6, 6))
    plt.imshow(imagen_coloreada)
    plt.title(titulo)
    plt.axis('off')
    plt.show()

# Función para mostrar jeroglíficos de referencia
def mostrar_jeroglificos_referencia(referencias):
    num_jeroglificos = len(referencias)
    fig, axes = plt.subplots(1, num_jeroglificos, figsize=(15, 5))
    fig.suptitle("Jeroglíficos de Referencia")
    for ax, (nombre, imagen) in zip(axes, referencias.items()):
        ax.imshow(imagen, cmap='gray')
        ax.set_title(nombre)
        ax.axis('off')
    plt.show()
