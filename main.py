import os
from preprocesamiento import cargar_jeroglificos_referencia, preprocesar_imagen
from morfologia import operaciones_morfologicas
from reconocimiento import etiquetar_componentes, reconocer_jeroglificos
from visualizacion import mostrar_imagen_binaria, mostrar_etiquetas_coloreadas, mostrar_jeroglificos_referencia

# Función principal
def principal(ruta_imagen, referencias):
    imagen_binaria = preprocesar_imagen(ruta_imagen)
    mostrar_imagen_binaria(imagen_binaria, f"Binarizada - {os.path.basename(ruta_imagen)}")
    imagen_procesada = operaciones_morfologicas(imagen_binaria)
    imagen_etiquetada, num_componentes = etiquetar_componentes(imagen_procesada)
    jeroglificos_reconocidos = reconocer_jeroglificos(imagen_etiquetada, referencias)
    
    # Mostrar los objetos detectados coloreados
    mostrar_etiquetas_coloreadas(imagen_etiquetada, imagen_binaria, f"Objetos detectados - {os.path.basename(ruta_imagen)}")
    
    # Ordenar y generar la cadena de salida
    jeroglificos_reconocidos.sort()
    salida = "".join(jeroglificos_reconocidos)
    return salida, num_componentes

# Ejemplo de uso
if __name__ == "__main__":
    referencias = cargar_jeroglificos_referencia()
    mostrar_jeroglificos_referencia(referencias)
    
    ejemplos = ["ejemplos/Ejemplo1.png", "ejemplos/Ejemplo2.png"]
    for ejemplo in ejemplos:
        try:
            resultado, num_componentes = principal(ejemplo, referencias)
            print(f"Resultado para {ejemplo}: {resultado} con {num_componentes} componentes reconocidos.")
        except (FileNotFoundError, RuntimeError) as e:
            print(e)
