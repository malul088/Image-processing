import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
import cv2
import os
import matplotlib.pyplot as plt

def initialize_kernel():
    kernel = np.array([
        [-1,  2,  1],
        [-2,  1, -3],
        [ 3,  0, -1]
    ], dtype=np.float32)
    
    return kernel

def get_image():
    image = np.array([
        [103, 102, 101, 100],
        [104, 103, 102, 101],
        [53,  52,  51,  50],
        [45,  53,  52,  51]
    ], dtype=np.uint8)
    return image

def cross_correlate_loop(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    out_h, out_w = get_w_h_out(w, h, kh)
    result = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            s = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    s += float(image[i+ki, j+kj] * kernel[ki,kj])
                result[i,j] = s
    return result

def cross_correlate_np(image, kernel):
    windows = sliding_window_view(image, kernel.shape)
    multiplied_windows = windows * kernel
    result = np.sum(multiplied_windows, axis=(2, 3))
    return result.astype(np.float32)

def cross_correlate_scipy(image, kernel):
    res = signal.correlate2d(image, kernel, mode='valid')
    return res.astype(np.float32)

def compare_cross_correlations():

    image = get_image()
    kernel = initialize_kernel()

    res_loop = cross_correlate_loop(image, kernel)
    res_np = cross_correlate_np(image, kernel)
    res_scipy = cross_correlate_scipy(image, kernel)
    
    is_loop_equal_np = np.allclose(res_loop, res_np)
    is_np_equal_scipy = np.allclose(res_np, res_scipy)
    
    return is_loop_equal_np and is_np_equal_scipy

def get_w_h_out(w_image, h_image, k):
    h_out = h_image - k + 1
    w_out = w_image - k + 1
    return h_out, w_out


def run_sobel(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image")
        return

    # א. המרה לצבעי אפור (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # המרת המטריצה ל-float32 כדי למנוע גלישת ערכים (Overflow) בזמן החישובים
    gray_f = gray.astype(np.float32)

    # הגדרת קרנלים של Sobel
    kernel_x = np.array([[-1, 0, 1], 
                         [-2, 0, 2], 
                         [-1, 0, 1]], dtype=np.float32)
    
    kernel_y = np.array([[-1, -2, -1], 
                         [ 0,  0,  0], 
                         [ 1,  2,  1]], dtype=np.float32)

    # חישוב Gx 
    gx = cross_correlate_np(gray_f, kernel_x)
    gx_abs = np.abs(gx)
    gx_norm = cv2.normalize(gx_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # חישוב Gy 
    gy = cross_correlate_np(gray_f, kernel_y)
    gy_abs = np.abs(gy)
    gy_norm = cv2.normalize(gy_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # חישוב  (G)
    magnitude = np.sqrt(gx**2 + gy**2)
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # שמירת התמונות
    base_name, ext = os.path.splitext(image_path)
    
    cv2.imwrite(f"{base_name}_grayscale{ext}", gray)
    cv2.imwrite(f"{base_name}_gx{ext}", gx_norm)
    cv2.imwrite(f"{base_name}_gy{ext}", gy_norm)
    cv2.imwrite(f"{base_name}_magnitude{ext}", mag_norm)

    print(f"Finished! Images saved in the same directory as {image_path}")
    return gray, gx_norm, gy_norm, mag_norm


def display_results(gray, gx, gy, magnitude):
    # יצירת לוח עם שורה אחת ו-4 עמודות
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # הגדרת כותרות לכל תמונה
    titles = ['Grayscale', 'Gx (Horizontal)', 'Gy (Vertical)', 'Magnitude (Edges)']
    images = [gray, gx, gy, magnitude]
    
    for i in range(4):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')  # העלמת הצירים (מספרי פיקסלים)
    
    plt.tight_layout() # סידור אוטומטי של הרווחים
    plt.show()


gray_img, gx_img, gy_img, mag_img = run_sobel("pic.jpg")
display_results(gray_img, gx_img, gy_img, mag_img)