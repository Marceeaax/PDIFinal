import os
import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import erosion, dilation, square, skeletonize

# Definir jeroglíficos de referencia
reference_hieroglyphs = {
    "ankh": "A",
    "wedjat": "J",
    "djed": "D",
    "scarab": "S",
    "was": "W",
    "akeht": "K"
}

# Cargar jeroglíficos de referencia
def load_reference_hieroglyphs():
    references = {}
    for name in reference_hieroglyphs.keys():
        image_path = f'jeroglificos/binarized/{name}.png'
        print(f"Intentando cargar {image_path}...")
        if not os.path.exists(image_path):
            print(f"Advertencia: No se encuentra el archivo de referencia: {image_path}")
            continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: No se pudo cargar la imagen de referencia: {image_path}")
            raise FileNotFoundError(f"No se pudo cargar la imagen de referencia: {image_path}")
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        references[name] = binary
        print(f"Imagen de referencia {name} cargada correctamente.")
    if not references:
        raise RuntimeError("No se pudieron cargar las imágenes de referencia.")
    return references

# Función para preprocesar la imagen
def preprocess_image(image_path):
    print(f"Intentando cargar la imagen de entrada: {image_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encuentra el archivo de entrada: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen de entrada: {image_path}")
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    print("Imagen de entrada cargada y binarizada correctamente.")
    return binary

# Función para aplicar erosión y dilatación
def morphological_operations(binary_image):
    print("Aplicando operaciones morfológicas...")
    se = square(3)
    eroded = erosion(binary_image, se)
    dilated = dilation(eroded, se)
    print("Operaciones morfológicas aplicadas correctamente.")
    return dilated

# Función para etiquetar componentes conectados
def label_components(binary_image):
    print("Etiquetando componentes conectados...")
    labeled_image = label(binary_image)
    print(f"{labeled_image.max()} componentes conectados etiquetados.")
    return labeled_image

# Función para extraer y comparar características
def recognize_hieroglyphs(labeled_image, references):
    print("Reconociendo jeroglíficos...")
    regions = regionprops(labeled_image)
    recognized_hieroglyphs = []
    
    for region in regions:
        skeleton = skeletonize(region.image)
        euler_number = region.euler_number
        print(f"Región con número de Euler {euler_number} detectada.")
        
        # Comparar con las características de los jeroglíficos de referencia
        matched = False
        for name, reference in references.items():
            ref_skeleton = skeletonize(reference)
            if np.array_equal(skeleton, ref_skeleton):
                recognized_hieroglyphs.append(reference_hieroglyphs[name])
                matched = True
                print(f"Jeroglífico {name} reconocido.")
                break
        
        if not matched:
            recognized_hieroglyphs.append("?")  # Si no hay coincidencia, marcar como desconocido
            print("Jeroglífico no reconocido.")
    
    return recognized_hieroglyphs

# Función principal
def main(image_path):
    references = load_reference_hieroglyphs()
    binary_image = preprocess_image(image_path)
    processed_image = morphological_operations(binary_image)
    labeled_image = label_components(processed_image)
    recognized_hieroglyphs = recognize_hieroglyphs(labeled_image, references)
    
    # Ordenar y generar la cadena de salida
    recognized_hieroglyphs.sort()
    output = "".join(recognized_hieroglyphs)
    return output

# Ejemplo de uso
if __name__ == "__main__":
    ejemplos = ["ejemplos/Ejemplo1.png", "ejemplos/Ejemplo2.png"]
    for ejemplo in ejemplos:
        try:
            result = main(ejemplo)
            print(f"Resultado para {ejemplo}: {result}")
        except (FileNotFoundError, RuntimeError) as e:
            print(e)
