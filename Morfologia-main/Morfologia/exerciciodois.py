import cv2
import numpy as np

# Carregando a imagem
imagem = cv2.imread('caminho/para/sua/imagem.jpg')

# Convertendo a imagem para escala de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplicando uma operação de limiarização para separar o branco do preto
_, imagem_limiarizada = cv2.threshold(imagem_cinza, 200, 255, cv2.THRESH_BINARY)

# Encontrando contornos na imagem limiarizada
contornos, _ = cv2.findContours(imagem_limiarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Encontrando o contorno do maior objeto (o retângulo central)
maior_contorno = max(contornos, key=cv2.contourArea)

# Criando uma imagem preta com o mesmo tamanho que a imagem original
imagem_resultante = np.zeros_like(imagem)

# Desenhando o contorno do maior objeto na imagem resultante
cv2.drawContours(imagem_resultante, [maior_contorno], 0, (255, 255, 255), thickness=cv2.FILLED)

# Exibindo a imagem resultante
cv2.imshow('Imagem Resultante', imagem_resultante)
cv2.waitKey(0)
cv2.destroyAllWindows()
