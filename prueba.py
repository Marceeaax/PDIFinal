import numpy as np
from skimage.measure import label

# Generar una matriz binaria aleatoria de tamaño 5x5
np.random.seed(0)  # Para reproducibilidad
binary_matrix = np.random.randint(0, 2, size=(5, 5))

# Mostrar la matriz binaria
print("Matriz Binaria:")
print(binary_matrix)

# Etiquetar los componentes conectados
labeled_matrix = label(binary_matrix, connectivity=2)

# Mostrar la matriz etiquetada
print("\nMatriz Etiquetada:")
print(labeled_matrix)