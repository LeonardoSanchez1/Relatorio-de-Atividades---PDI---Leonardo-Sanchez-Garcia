import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_morphological_operation(image, operation, kernel):
    if operation == 'erode':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'dilate':
        return cv2.dilate(image, kernel, iterations=1)
    else:
        raise ValueError("Operação morfológica inválida. Escolha 'erode' ou 'dilate'.")

if __name__ == "__main__":
    # Carregue a imagem de exemplo (substitua pelo caminho da sua imagem)
    image = cv2.imread('fingerprint.tif', cv2.IMREAD_GRAYSCALE)

    # Defina os elementos estruturantes
    kernel1 = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

    kernel2 = np.ones((3, 3), np.uint8)

    kernel3 = np.ones((7, 1), np.uint8)

    kernel4 = np.array([[0, 0, 1, 0, 0],
                        [0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0]], dtype=np.uint8)

    # Aplique a erosão e dilatação usando os elementos estruturantes especificados
    result_erode1 = apply_morphological_operation(image, 'erode', kernel1)
    result_dilate1 = apply_morphological_operation(image, 'dilate', kernel1)

    result_erode2 = apply_morphological_operation(image, 'erode', kernel2)
    result_dilate2 = apply_morphological_operation(image, 'dilate', kernel2)

    result_erode3 = apply_morphological_operation(image, 'erode', kernel3)
    result_dilate3 = apply_morphological_operation(image, 'dilate', kernel3)

    result_erode4 = apply_morphological_operation(image, 'erode', kernel4)
    result_dilate4 = apply_morphological_operation(image, 'dilate', kernel4)

    # Exiba as imagens original e após aplicar erosão/dilatação com diferentes elementos estruturantes usando o Matplotlib
    plt.figure(figsize=(10, 8))

    plt.subplot(4, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(4, 3, 2)
    plt.imshow(result_erode1, cmap='gray')
    plt.title('Erosão')

    plt.subplot(4, 3, 3)
    plt.imshow(result_dilate1, cmap='gray')
    plt.title('Dilatação')

    plt.subplot(4, 3, 4)
    plt.imshow(image, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(4, 3, 5)
    plt.imshow(result_erode2, cmap='gray')
    plt.title('Erosão')

    plt.subplot(4, 3, 6)
    plt.imshow(result_dilate2, cmap='gray')
    plt.title('Dilatação')

    plt.subplot(4, 3, 7)
    plt.imshow(image, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(4, 3, 8)
    plt.imshow(result_erode3, cmap='gray')
    plt.title('Erosão')

    plt.subplot(4, 3, 9)
    plt.imshow(result_dilate3, cmap='gray')
    plt.title('Dilatação')

    plt.subplot(4, 3, 10)
    plt.imshow(image, cmap='gray')
    plt.title('Imagem Original')

    plt.subplot(4, 3, 11)
    plt.imshow(result_erode4, cmap='gray')
    plt.title('Erosão')

    plt.subplot(4, 3, 12)
    plt.imshow(result_dilate4, cmap='gray')
    plt.title('Dilatação')

    plt.tight_layout()
    plt.show()
