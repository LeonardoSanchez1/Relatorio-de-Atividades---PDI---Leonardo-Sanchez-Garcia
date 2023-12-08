import cv2
import numpy as np

def extract_edges(image, iterations_dilation, iterations_erosion):
    # Converta a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplique uma suavização para reduzir o ruído
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Aplique a operação de dilatação para realçar as bordas
    kernel_dilation = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(blurred_image, kernel_dilation, iterations=iterations_dilation)

    # Aplique a operação de erosão para refinar as bordas
    kernel_erosion = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(dilated_image, kernel_erosion, iterations=iterations_erosion)

    # Calcule as bordas subtraindo a imagem erodida da imagem dilatada
    edges = cv2.absdiff(dilated_image, eroded_image)

    # Aplique uma limiarização para destacar as bordas
    _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)

    return edges

if __name__ == "__main__":
    # Carregue a imagem de exemplo (substitua pelo caminho da sua imagem)
    image_h = cv2.imread('rosto_perfil.tif')

    # Extraia as bordas
    edges = extract_edges(image_h, iterations_dilation=1, iterations_erosion=1)

    # Exiba as imagens original e as bordas
    cv2.imshow('Original', image_h)
    cv2.imshow('Borda', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
