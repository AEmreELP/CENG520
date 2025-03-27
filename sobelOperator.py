import cv2
import numpy as np
import matplotlib.pyplot as plt


def sobel_operator(image_path):
    # Görüntüyü yükle ve gri tonlamaya çevir
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel kernelleri tanımla
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Manuel olarak Sobel operatörünü uygula
    # Konvolüsyon işlemi
    gradient_x = cv2.filter2D(gray, -1, sobel_x)
    gradient_y = cv2.filter2D(gray, -1, sobel_y)

    # Gradyanların büyüklüğünü hesapla
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)

    # OpenCV'nin yerleşik Sobel fonksiyonu ile karşılaştırma
    opencv_gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    opencv_gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    opencv_magnitude = np.sqrt(np.square(opencv_gradient_x) + np.square(opencv_gradient_y))
    opencv_magnitude = np.uint8(opencv_magnitude / opencv_magnitude.max() * 255)

    # Sonuçları göster
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Orijinal Gri Görüntü')

    plt.subplot(2, 3, 2)
    plt.imshow(gradient_x, cmap='gray')
    plt.title('Manuel Sobel X')

    plt.subplot(2, 3, 3)
    plt.imshow(gradient_y, cmap='gray')
    plt.title('Manuel Sobel Y')

    plt.subplot(2, 3, 4)
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.title('Manuel Sobel Büyüklüğü')

    plt.subplot(2, 3, 5)
    plt.imshow(opencv_magnitude, cmap='gray')
    plt.title('OpenCV Sobel Büyüklüğü')

    plt.tight_layout()
    plt.show()

    return gradient_magnitude, opencv_magnitude


# Kullanım örneği
if __name__ == "__main__":
    # Görüntü yolunu buraya girin
    image_path = "image.jpg"
    manuel_sobel, opencv_sobel = sobel_operator(image_path)
