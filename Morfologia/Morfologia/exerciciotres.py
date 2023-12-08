import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image):
    # Converta a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplique uma limiarização para binarizar a imagem
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    # Aplique a erosão para remover manchas brancas ao redor do retângulo
    eroded_image = cv2.erode(binary_image, np.ones((50, 50), np.uint8), iterations=1)

    # Aplique a dilatação para ampliar o retângulo branco central
    dilated_image = cv2.dilate(eroded_image, np.ones((30, 30), np.uint8), iterations=1)

    # Aplique o fechamento para remover manchas pretas dentro do retângulo
    closed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, np.ones((60, 60), np.uint8))

    return closed_image

if __name__ == "__main__":
    # Carregue a imagem de exemplo (substitua pelo caminho da sua imagem)
    image_f = cv2.imread('noise_rectangle.tif')

    # Processar a imagem para manter apenas o retângulo branco ao centro
    processed_image = process_image(image_f)

    # Redimensione a imagem para um tamanho menor
    processed_image_resized = cv2.resize(processed_image, (400, 400))

    # Exiba as imagens original e processada com janelas do Matplotlib de tamanho médio
    plt.figure(figsize=(8, 8))
    plt.subplot(121), plt.imshow(cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(122), plt.imshow(processed_image_resized, cmap='gray'), plt.title('Processed')
    plt.show()
