import os  # Importa el módulo os para interactuar con el sistema operativo
from preprocesamiento import cargar_jeroglificos_referencia, preprocesar_imagen  # Importa funciones del módulo preprocesamiento
from morfologia import operaciones_morfologicas  # Importa funciones del módulo morfología
from reconocimiento import etiquetar_componentes, reconocer_jeroglificos  # Importa funciones del módulo reconocimiento
from visualizacion import mostrar_imagen_binaria, mostrar_etiquetas_coloreadas, mostrar_jeroglificos_referencia  # Importa funciones del módulo visualización

# Función principal para procesar y reconocer jeroglíficos en una imagen dada
def principal(ruta_imagen, referencias):
    # Preprocesa la imagen de entrada
    imagen_binaria = preprocesar_imagen(ruta_imagen)
    # Muestra la imagen binarizada
    mostrar_imagen_binaria(imagen_binaria, f"Binarizada - {os.path.basename(ruta_imagen)}")
    
    # Aplica operaciones morfológicas a la imagen binarizada
    imagen_procesada = operaciones_morfologicas(imagen_binaria)
    
    # Etiqueta los componentes conectados en la imagen procesada
    imagen_etiquetada, num_componentes = etiquetar_componentes(imagen_procesada)
    
    # Reconoce los jeroglíficos en la imagen etiquetada
    jeroglificos_reconocidos = reconocer_jeroglificos(imagen_etiquetada, referencias)
    
    # Muestra los objetos detectados coloreados
    mostrar_etiquetas_coloreadas(imagen_etiquetada, imagen_binaria, f"Objetos detectados - {os.path.basename(ruta_imagen)}")
    
    # Ordena los jeroglíficos reconocidos y genera la cadena de salida
    jeroglificos_reconocidos.sort()
    salida = "".join(jeroglificos_reconocidos)
    
    return salida, num_componentes

# Bloque de código que se ejecuta solo si el archivo se ejecuta directamente (no se importa como módulo)
if __name__ == "__main__":
    # Carga las imágenes de referencia de jeroglíficos
    referencias = cargar_jeroglificos_referencia()
    # Muestra las imágenes de referencia cargadas
    mostrar_jeroglificos_referencia(referencias)
    
    # Lista de ejemplos de imágenes a procesar
    ejemplos = ["ejemplos/Ejemplo6.png", "ejemplos/Ejemplo1.png"]
    
    # Itera sobre cada imagen de ejemplo y procesa cada una
    for ejemplo in ejemplos:
        try:
            # Llama a la función principal para procesar la imagen de ejemplo y obtener los resultados
            resultado, num_componentes = principal(ejemplo, referencias)
            # Imprime los resultados obtenidos
            print(f"Resultado para {ejemplo}: {resultado} con {num_componentes} componentes reconocidos.")
        except (FileNotFoundError, RuntimeError) as e:
            # Captura y muestra cualquier error que ocurra durante el procesamiento
            print(e)
