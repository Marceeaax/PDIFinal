from skimage.morphology import square, closing

# Función para aplicar erosión y dilatación
def operaciones_morfologicas(imagen_binaria):
    print("Aplicando operaciones morfológicas...")
    se = square(3)
    cerrada = closing(imagen_binaria, se)
    print("Operaciones morfológicas aplicadas correctamente.")
    return cerrada
