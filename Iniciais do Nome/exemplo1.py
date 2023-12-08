import numpy as np
import matplotlib.pyplot as plt

colunas = 50
linhas = 25
image_matrix = np.zeros([linhas, colunas])
print(image_matrix.shape)

#Acessando um pixel
#image_matrix[0:,2:] = 255
#Acessando uma linha (Primeira linha da matrix)
#image_matrix[0,:] = 255
#Acessando mais de uma linha (Quarta e quinta linha da matrix)
#image_matrix[:,:] = 255
#Acessando uma coluna (Ultima coluna)
image_matrix[5:14,10] = 120
image_matrix[13,10:14] = 120

image_matrix[5:9,18] = 120
image_matrix[5,18:22] = 120
image_matrix[9,18:22] = 120
image_matrix[9:13,21] = 120
image_matrix[13,18:22] = 120

image_matrix[5,26:31] = 120
image_matrix[5:14,26] = 120
image_matrix[13,26:31] = 120
image_matrix[9:13,30] = 120
image_matrix[9,28:31] = 120

plt.imshow(image_matrix, cmap='gray')
plt.show()


