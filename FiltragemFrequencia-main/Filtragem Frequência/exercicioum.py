import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Crie e visualize uma imagem simples - quadrado branco sobre fundo preto.
image = np.zeros((512, 512), dtype=np.uint8)
cv2.rectangle(image, (100, 100), (412, 412), 255, -1)

# 2. Calcule e visualize o espectro de Fourier (amplitudes).
f_transform = np.fft.fft2(image)
f_transform_shifted = np.fft.fftshift(f_transform)
magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

# 3. Calcule e visualize o espectro de Fourier (fases).
phase_spectrum = np.angle(f_transform_shifted)

# 4. Obtenha e visualize o espectro de Fourier centralizado.
magnitude_spectrum_centered = np.log(np.abs(f_transform) + 1)

# Visualize a imagem original e os espectros de Fourier centralizados.
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Imagem Original'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Espectro de Fourier (Amplitudes)'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(phase_spectrum, cmap='gray')
plt.title('Espectro de Fourier (Fases)'), plt.xticks([]), plt.yticks([])
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(magnitude_spectrum_centered, cmap='gray')
plt.title('Espectro de Fourier (Centralizada)'), plt.xticks([]), plt.yticks([])
plt.show()


# 5. Aplique uma rotação de 40º na imagem e repita os passos 2-4.
rotation_matrix = cv2.getRotationMatrix2D((256, 256), 40, 1)
rotated_image = cv2.warpAffine(image, rotation_matrix, (512, 512))
f_transform_rotated = np.fft.fft2(rotated_image)
f_transform_rotated_shifted = np.fft.fftshift(f_transform_rotated)
magnitude_spectrum_rotated = np.log(np.abs(f_transform_rotated_shifted) + 1)
phase_spectrum_rotated = np.angle(f_transform_rotated_shifted)

# Obtendo e visualizando o espectro de Fourier centralizado da imagem rotacionada.
f_transform_rotated_centered = np.fft.fft2(rotated_image)
magnitude_spectrum_rotated_centered = np.log(np.abs(np.fft.fftshift(f_transform_rotated_centered)) + 1)
phase_spectrum_rotated_centered = np.angle(np.fft.fftshift(f_transform_rotated_centered))

# Visualizar os resultados da imagem rotacionada e espectros de Fourier centralizados.
plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(rotated_image, cmap='gray')
plt.title('Imagem Rotacionada'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(magnitude_spectrum_rotated, cmap='gray')
plt.title('Espectro de Fourier da Imagem Rotacionada'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(phase_spectrum_rotated, cmap='gray')
plt.title('Fases do Espectro da Imagem Rotacionada'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(magnitude_spectrum_rotated_centered, cmap='gray')
plt.title('Espectro de Fourier Centralizado da Imagem Rotacionada'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(phase_spectrum_rotated_centered, cmap='gray')
plt.title('Fases do Espectro Centralizado da Imagem Rotacionada'), plt.xticks([]), plt.yticks([])

# 6. Aplique uma translação nos eixos x e y na imagem e repita os passos 2-4.
translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
translated_image = cv2.warpAffine(image, translation_matrix, (512, 512))
f_transform_translated = np.fft.fft2(translated_image)
f_transform_translated_shifted = np.fft.fftshift(f_transform_translated)
magnitude_spectrum_translated = np.log(np.abs(f_transform_translated_shifted) + 1)
phase_spectrum_translated = np.angle(f_transform_translated_shifted)

# Obtendo e visualizando o espectro de Fourier centralizado da imagem transladada.
f_transform_translated_centered = np.fft.fft2(translated_image)
magnitude_spectrum_translated_centered = np.log(np.abs(np.fft.fftshift(f_transform_translated_centered)) + 1)
phase_spectrum_translated_centered = np.angle(np.fft.fftshift(f_transform_translated_centered))

# Visualizar os resultados da imagem transladada e espectros de Fourier centralizados.
plt.subplot(236), plt.imshow(translated_image, cmap='gray')
plt.title('Imagem Transladada'), plt.xticks([]), plt.yticks([])

plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(magnitude_spectrum_translated, cmap='gray')
plt.title('Espectro de Fourier da Imagem Transladada'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(phase_spectrum_translated, cmap='gray')
plt.title('Fases do Espectro da Imagem Transladada'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(magnitude_spectrum_translated_centered, cmap='gray')
plt.title('Espectro de Fourier Centralizado da Imagem Transladada'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(phase_spectrum_translated_centered, cmap='gray')
plt.title('Fases do Espectro Centralizado da Imagem Transladada'), plt.xticks([]), plt.yticks([])

# 7. Aplique um zoom na imagem e repita os passos 2-4.
zoomed_image = cv2.resize(image, (256, 256))
f_transform_zoomed = np.fft.fft2(zoomed_image, (512, 512))
f_transform_zoomed_shifted = np.fft.fftshift(f_transform_zoomed)
magnitude_spectrum_zoomed = np.log(np.abs(f_transform_zoomed_shifted) + 1)
phase_spectrum_zoomed = np.angle(f_transform_zoomed_shifted)

# Obtendo e visualizando o espectro de Fourier centralizado da imagem ampliada.
f_transform_zoomed_centered = np.fft.fft2(zoomed_image, (512, 512))
magnitude_spectrum_zoomed_centered = np.log(np.abs(np.fft.fftshift(f_transform_zoomed_centered)) + 1)
phase_spectrum_zoomed_centered = np.angle(np.fft.fftshift(f_transform_zoomed_centered))

# Visualizar os resultados da imagem ampliada e espectros de Fourier centralizados.
plt.figure(figsize=(15, 10))
plt.subplot(231), plt.imshow(zoomed_image, cmap='gray')
plt.title('Imagem Ampliada'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(magnitude_spectrum_zoomed, cmap='gray')
plt.title('Espectro de Fourier da Imagem Ampliada'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(phase_spectrum_zoomed, cmap='gray')
plt.title('Fases do Espectro da Imagem Ampliada'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(magnitude_spectrum_zoomed_centered, cmap='gray')
plt.title('Espectro de Fourier Centralizado da Imagem Ampliada'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(phase_spectrum_zoomed_centered, cmap='gray')
plt.title('Fases do Espectro Centralizado da Imagem Ampliada'), plt.xticks([]), plt.yticks([])

plt.show()
