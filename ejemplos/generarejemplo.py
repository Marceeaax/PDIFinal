import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt

# Directorio de entrada
input_dir = '../jeroglificos/binarized'

# Lista de archivos de jeroglíficos
jeroglificos = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# Contar el número de ejemplos existentes en el directorio de salida
existing_examples = len([f for f in os.listdir('.') if f.startswith('Ejemplo') and f.endswith('.png')])
start_index = existing_examples + 1

# Generar imágenes de ejemplo
num_examples = 2
example_size = (300, 300)

# Función para verificar la superposición considerando solo píxeles de primer plano
def check_overlap(existing_positions, new_position, jeroglifico):
    for pos in existing_positions:
        if (new_position[0] < pos[0] + pos[2] and
            new_position[0] + jeroglifico.shape[1] > pos[0] and
            new_position[1] < pos[1] + pos[3] and
            new_position[1] + jeroglifico.shape[0] > pos[1]):
            # Verificar superposición de píxeles de primer plano
            for x in range(jeroglifico.shape[1]):
                for y in range(jeroglifico.shape[0]):
                    if jeroglifico[y, x] == 0:  # Solo considerar píxeles de primer plano
                        if example_image[new_position[1] + y, new_position[0] + x] == 0:
                            return True
    return False

# Función para extraer solo los píxeles de primer plano
def extract_foreground(jeroglifico):
    foreground_pixels = np.argwhere(jeroglifico == 0)
    return foreground_pixels

for i in range(start_index, start_index + num_examples):
    # Crear una imagen en blanco (fondo blanco)
    example_image = np.ones(example_size, dtype=np.uint8) * 255
    existing_positions = []
    
    # Elegir un número aleatorio de jeroglíficos para pegar
    num_jeroglificos = random.randint(1, 5)
    
    for _ in range(num_jeroglificos):
        # Elegir un jeroglífico al azar
        jeroglifico_file = random.choice(jeroglificos)
        jeroglifico_path = os.path.join(input_dir, jeroglifico_file)
        
        # Cargar el jeroglífico binarizado
        jeroglifico = cv2.imread(jeroglifico_path, cv2.IMREAD_GRAYSCALE)

        # Redimensionar el jeroglífico a un tamaño más pequeño y rotarlo
        scale_factor = random.uniform(0.5, 1.5)
        jeroglifico = cv2.resize(jeroglifico, (int(jeroglifico.shape[1] * scale_factor), int(jeroglifico.shape[0] * scale_factor)), interpolation=cv2.INTER_NEAREST)
        angle = random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((jeroglifico.shape[1]//2, jeroglifico.shape[0]//2), angle, 1)
        jeroglifico = cv2.warpAffine(jeroglifico, M, (jeroglifico.shape[1], jeroglifico.shape[0]), borderValue=(255))

        # Extraer píxeles de primer plano
        foreground_pixels = extract_foreground(jeroglifico)

        # Verificar si el jeroglífico cabe en la imagen de ejemplo
        if jeroglifico.shape[0] > example_size[0] or jeroglifico.shape[1] > example_size[1]:
            continue  # Saltar este jeroglífico si es demasiado grande

        # Elegir una posición aleatoria para pegar el jeroglífico sin superposición y sin estar pegado a los bordes
        placed = False
        attempts = 0
        while not placed and attempts < 10:  # Limitar el número de intentos para evitar bucles infinitos
            x_offset = random.randint(10, example_size[1] - jeroglifico.shape[1] - 10)
            y_offset = random.randint(10, example_size[0] - jeroglifico.shape[0] - 10)
            if not check_overlap(existing_positions, (x_offset, y_offset, jeroglifico.shape[1], jeroglifico.shape[0]), jeroglifico):
                # Pegar el jeroglífico en la imagen de ejemplo sin cambiar el fondo blanco
                for pixel in foreground_pixels:
                    if 0 <= y_offset + pixel[0] < example_size[0] and 0 <= x_offset + pixel[1] < example_size[1]:
                        example_image[y_offset + pixel[0], x_offset + pixel[1]] = 0
                existing_positions.append((x_offset, y_offset, jeroglifico.shape[1], jeroglifico.shape[0]))
                placed = True
            attempts += 1
    
    # Guardar la imagen de ejemplo
    output_path = f'Ejemplo{i}.png'
    cv2.imwrite(output_path, example_image)  # Guardar como imagen binaria (0 y 255)

# Mostrar algunas de las imágenes de ejemplo generadas
for i in range(start_index, start_index + 3):
    example_image = cv2.imread(f'Ejemplo{i}.png', cv2.IMREAD_GRAYSCALE)
    plt.figure(figsize=(5, 5))
    plt.title(f'Ejemplo {i}')
    plt.imshow(example_image, cmap='gray')
    plt.show()
