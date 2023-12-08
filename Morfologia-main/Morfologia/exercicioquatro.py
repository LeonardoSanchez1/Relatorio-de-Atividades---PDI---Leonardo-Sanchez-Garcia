import cv2
import numpy as np

def improve_text(image, dilation_kernel_size, iterations_dilation):
    # Dilatação leve para melhorar as letras brancas
    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    improved_image = cv2.dilate(image, dilation_kernel, iterations=iterations_dilation)

    return improved_image

if __name__ == "__main__":
    # Carregue a imagem de exemplo (substitua pelo caminho da sua imagem)
    image_g = cv2.imread('text_gaps.tif', cv2.IMREAD_GRAYSCALE)

    # Aplique a dilatação para melhorar as letras brancas
    improved_image = improve_text(image_g, dilation_kernel_size=3, iterations_dilation=1)

    # Exiba as imagens original e aprimorada
    cv2.imshow('Original', image_g)
    cv2.imshow('Improved', improved_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
