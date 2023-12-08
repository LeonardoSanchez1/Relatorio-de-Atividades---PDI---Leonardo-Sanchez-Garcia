import numpy as np
import matplotlib.pyplot as plt

# Dimensões da imagem
width = 400
height = 400

# Cria uma matriz de uns para representar o fundo branco
image = np.ones((height, width))

# Defina os parâmetros do quadrado no meio
square_size = 100
x_center = width // 2
y_center = height // 2

# Preenche o quadrado com a função sinc
for x in range(x_center - square_size // 2, x_center + square_size // 2):
    for y in range(y_center - square_size // 2, y_center + square_size // 2):
        # Calcula o valor da função sinc em (x, y)
        value = np.sinc((x - x_center) / square_size) * np.sinc((y - y_center) / square_size)
        # Define o valor na matriz da imagem como 1 - value para torná-lo preto
        image[y, x] = 1 - value

# Plota a imagem
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.show()
