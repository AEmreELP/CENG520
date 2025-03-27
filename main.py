import cv2
import numpy as np
import math

def compute_kitchen_rosenfeld_cornerness(gray):
    """
    Kitchen-Rosenfeld benzeri köşe ölçüsünü hesaplar.
    Bu örnekte, içerikteki açık formüllerden yola çıkarak basitçe Ix^2 + Iy^2 kullanılmıştır.
    Gerçek Hayatta, Kitchen-Rosenfeld metodu literatürde farklı şekillerde uygulanabilir.
    """
    # Sobel operatörü ile x ve y yönlerinde türev hesaplama
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
    Ixy = cv2.Sobel(Ix, cv2.CV_64F, 0, 1, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)
    Iyx = cv2.Sobel(Iy, cv2.CV_64F, 1, 0, ksize=3)
    Ix = cv2.convertScaleAbs(Ix)
    Iy = cv2.convertScaleAbs(Iy)
    Ixx = cv2.convertScaleAbs(Ixx)
    Ixy = cv2.convertScaleAbs(Ixy)
    Iyy = cv2.convertScaleAbs(Iyy)
    Iyx = cv2.convertScaleAbs(Iyx)
    print(Ix.shape)
    print(Iy.shape)
    print(Ixx.shape)
    print(Ixy.shape)
    print(Iyy.shape)
    print(Iyx.shape)

    denom = np.sqrt(Ix ** 2 + Iy ** 2)


    # vektör v = [-Iy, Ix]
    V = np.array([-Iy, Ix])
    print(V.shape)
    # Hessian matris H
    H = np.array([[Ixx, Ixy], [Iyx, Iyy]])
    print(H.shape)
    out=np.matmul(H, V)
    print(out)
    VT = V.reshape(2, -1)

    print(V)
    print(H)

    print(VT.shape)

    # temp = np.dot(V,H)
    # val = np.dot(temp,VT)
    # print(val)


    # Sonuç (denklemdeki 1/(sqrt(Ix^2 + Iy^2))^2 = 1/(Ix^2 + Iy^2))
    # K = val / (denom ** 2)
    return 0



def non_maximum_suppression(cornerness, window_size=3):
    """
    Belirtilen pencere boyutu (window_size) içerisinde non-maxima suppression uygulaması.
    Her piksel için, pencere içinde en büyük değere sahip değilse değeri 0 yapılır.
    """
    half_w = window_size // 2
    suppressed = np.zeros_like(cornerness)
    rows, cols = cornerness.shape
    for i in range(half_w, rows - half_w):
        for j in range(half_w, cols - half_w):
            window = cornerness[i - half_w:i + half_w + 1, j - half_w:j + half_w + 1]
            # Eğer merkeze ait değer pencerenin maksimumu ise değeri koru.
            if cornerness[i, j] == np.max(window):
                suppressed[i, j] = cornerness[i, j]
    return suppressed


def mark_corners_on_image(image, corners, thresh):
    """
    Belirtilen eşik değerinin üzerindeki köşe noktalarını kırmızı dairelerle işaretler.
    """
    marked_image = image.copy()
    # Köşeyi işaretlemek için eşik değeri kullanılır
    corners_thresh = np.where(corners > thresh)
    for pt in zip(*corners_thresh[::-1]):  # (x,y) formatında
        cv2.circle(marked_image, pt, radius=3, color=(0, 0, 255), thickness=-1)
    return marked_image


def main():
    # Görüntüyü yükle
    image = cv2.imread('image.jpg')
    if image is None:
        print("Görüntü yüklenemedi! Dosya yolunu kontrol edin.")
        return
    # Gri tonlamalı görüntüye çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Köşe ölçüsünü hesapla
    cornerness = compute_kitchen_rosenfeld_cornerness(gray)

    # # Kullanıcı tanımlı pencere boyutunu belirle (örneğin: 3x3)
    # window_size = 3  # Bu değeri isteğinize göre değiştirebilirsiniz.
    # suppressed = non_maximum_suppression(cornerness, window_size=window_size)
    #
    # # Eşik değerini belirle. Bu değer deneysel olarak ayarlanmalıdır.
    # thresh = 1e6  # Örneğin, 1e6 gibi bir eşik değeri; görüntünüze bağlı olarak değiştirin.
    # marked_image = mark_corners_on_image(image, suppressed, thresh)

    # Sonuçları göster
    # cv2.imshow("Original Image", image)
    # cv2.imshow("Cornerness", cv2.normalize(cornerness, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    # cv2.imshow("Suppressed Corners", cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    # cv2.imshow("Corners on Image", marked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
