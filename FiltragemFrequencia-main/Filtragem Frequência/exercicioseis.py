import cv2
import numpy as np
from scipy import ndimage

# Carregue a imagem
imagem = cv2.imread('sinc_original_menor.tif', cv2.IMREAD_GRAYSCALE)

# Defina as frequências de corte inferior e superior
f1 = 30  # Frequência de corte inferior
f2 = 70  # Frequência de corte superior

# Crie um filtro passa-banda usando um kernel Gaussiano
filtro_passa_banda = ndimage.gaussian_filter(imagem, sigma=f2/5) - ndimage.gaussian_filter(imagem, sigma=f1/5)

# Normalize a imagem resultante para valores entre 0 e 255
imagem_filtrada = (filtro_passa_banda - np.min(filtro_passa_banda)) / (np.max(filtro_passa_banda) - np.min(filtro_passa_banda)) * 255

# Converte para tipo uint8
imagem_filtrada = imagem_filtrada.astype(np.uint8)

# Exiba a imagem original e a imagem filtrada
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Imagem Filtrada', imagem_filtrada)

cv2.waitKey(0)
cv2.destroyAllWindows()
