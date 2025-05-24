import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

start = datetime.now()


def display_image(image, figsize=(8, 8), title=None):
    """Exibe uma imagem com matplotlib em RGB."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


# Carrega a imagem
image_original = cv2.imread('foto.jpeg')

# Rotaciona a imagem 90 graus no sentido anti-horário
image_original = cv2.rotate(image_original, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imwrite('image_original.jpg', image_original)

image = image_original.copy()
display_image(image, figsize=(8, 8), title='Imagem Original')

# Detecta bordas
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image(gray, figsize=(8, 8), title='Imagem Grayscale')
cv2.imwrite('gray.jpg', gray)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
display_image(blurred, figsize=(8, 8), title='Imagem Desfocada')
cv2.imwrite('blur.jpg', blurred)

edged = cv2.Canny(blurred, 75, 200)
display_image(edged, figsize=(8, 8), title='Imagem Com Filtro Canny')
cv2.imwrite('edged.jpg', edged)

# Converte imagens de BGR para RGB para exibição com matplotlib
img1_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
img3_rgb = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
img4_rgb = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

axs[0].imshow(img1_rgb)
axs[1].imshow(img2_rgb)
axs[2].imshow(img3_rgb)
axs[3].imshow(img4_rgb)

for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('preprocessing.jpg', bbox_inches='tight', pad_inches=0)
plt.show()

# Encontra os contornos
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# Ordena os contornos pela área e pega os 5 maiores
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

doc_cnt = None

for c in contours:
    # Aproxima o contorno para um polígono
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)

    # Verifica se o polígono tem 4 vértices (possível documento)
    if len(approx) == 4:
        doc_cnt = approx
        break

if doc_cnt is None:
    raise ValueError('Documento não encontrado nos contornos.')

# Reorganiza os pontos em formato float32
points = doc_cnt.reshape(4, 2).astype('float32')

# Desenha círculos vermelhos nos pontos identificados
for point in points:
    center = (int(point[0]), int(point[1]))
    cv2.circle(image, center, radius=5, color=(0, 0, 255), thickness=-1)

display_image(image, figsize=(8, 8), title='Imagem Original Com Cantos Identificados')
cv2.imwrite('points.jpg', image)

# Desenha o contorno da tabela em azul
cv2.drawContours(image, [doc_cnt], -1, (255, 0, 0), 2)
display_image(image, figsize=(8, 8), title='Imagem Original Com Contorno na Tabela')
cv2.imwrite('contour.jpg', image)

# Ordena os pontos na ordem: topo-esquerdo, topo-direito, baixo-direito, baixo-esquerdo
ordered_points = np.zeros((4, 2), dtype='float32')

s = points.sum(axis=1)
ordered_points[0] = points[np.argmin(s)]
ordered_points[2] = points[np.argmax(s)]

diff = np.diff(points, axis=1)
ordered_points[1] = points[np.argmin(diff)]
ordered_points[3] = points[np.argmax(diff)]

(tl, tr, br, bl) = ordered_points

# Calcula a largura máxima da nova imagem
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
max_width = max(int(widthA), int(widthB))

# Calcula a altura máxima da nova imagem
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
max_height = max(int(heightA), int(heightB))

# Define os pontos de destino para a transformação perspectiva
destination = np.array([
    [0, 0],
    [max_width - 1, 0],
    [max_width - 1, max_height - 1],
    [0, max_height - 1]
], dtype='float32')

# Calcula a matriz de transformação perspectiva
M = cv2.getPerspectiveTransform(ordered_points, destination)

# Aplica a transformação perspectiva para obter a imagem retificada
warped = cv2.warpPerspective(image_original.copy(), M, (max_width, max_height))

display_image(image_original, figsize=(8, 8), title='Imagem Original')
display_image(warped, figsize=(8, 8), title='Tabela Extraída')

cv2.imwrite('warped.jpg', warped)

print(f'Tempo decorrido: {datetime.now() - start}')
