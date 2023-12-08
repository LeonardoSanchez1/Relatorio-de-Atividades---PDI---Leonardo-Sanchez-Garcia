import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
imagem = cv2.imread('fractured_spine.tif', 0)

# Aplicar a equalização do histograma
imagem_equalizada = cv2.equalizeHist(imagem)

# Exibir a imagem original e a imagem equalizada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagem, cmap='gray')
plt.title('Imagem Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagem_equalizada, cmap='gray')
plt.title('Imagem Equalizada')
plt.axis('off')

plt.tight_layout()
plt.show()
