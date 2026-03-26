import math
import numpy as np

def warp_image(image: np.ndarray,
               angle_deg: float,
               scale_x: float,
               scale_y: float) -> np.ndarray:

    H, W, C = image.shape

    # TODO:

    # 1. Compute center (cx, cy)
    cx, cy = W/2.0, H/2.0

    # 2. Build rotation matrix
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotation = np.array([[cos_t, -sin_t, 0],[sin_t, cos_t, 0],[0,0,1]])

    # 3. Build scaling matrix
    scaling=np.array([[scale_x,0,0],[0,scale_y,0],[0,0,1]])

    # 4. Compose full affine matrix
    translation = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    translationBack = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    affine = translation @ rotation @ scaling @ translationBack

    # 5. Compute inverse
    inverse = np.linalg.inv(affine)

    # 6. For each output pixel:
    output = np.zeros_like(image)
    for r in range(H):
        for c in range(W):
            # -map center coordinate backward
            x_dst = c + 0.5
            y_dst = r + 0.5
            src = inverse @ np.array([x_dst, y_dst, 1.0])
            x_src = src[0]
            y_src = src[1]
            r_src = y_src - 0.5
            c_src = x_src - 0.5

            # -interpolate
            newColor = bilinear_interpolation(c_src, r_src, H, W, image)
            output[r, c] = newColor

    # 7. Return output image
    return  output


def bilinear_interpolation(c_src, r_src, H, W, image):
    c0 = int(np.floor(c_src))
    r0 = int(np.floor(r_src))
    c1 = c0 + 1
    r1 = r0 + 1

    if 0 <= r0 < H - 1 and 0 <= c0 < W - 1:
        alpha = c_src - c0
        beta = r_src - r0
        I00 = image[r0, c0]
        I01 = image[r0, c1]
        I10 = image[r1, c0]
        I11 = image[r1, c1]

        val = (1-alpha)*(1-beta)*I00 + alpha*(1-beta)*I01 + (1-alpha)*beta*I10 + alpha*beta*I11
        return val
    else: return [0,0,0]


def warp_image_numpy(image: np.ndarray,
                                   angle_deg: float,
                                   scale_x: float,
                                   scale_y: float) -> np.ndarray:
    H, W, C = image.shape
    cx, cy = W / 2.0, H / 2.0
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    scaling = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
    translation = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    translationBack = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    affine = translation @ rotation @ scaling @ translationBack
    inv_affine = np.linalg.inv(affine)

    # 2. יצירת רשת קואורדינטות (Grid) עבור כל הפיקסלים ביעד
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    # הפיכה לוקטורים הומוגניים (x, y, 1) בבת אחת
    dst_coords = np.stack([cols.flatten() + 0.5,
                           rows.flatten() + 0.5,
                           np.ones(H * W)])

    # 3. חישוב קואורדינטות המקור (Backward Mapping)
    src_coords = inv_affine @ dst_coords
    src_x = src_coords[0, :] - 0.5
    src_y = src_coords[1, :] - 0.5

    # 4. הכנה לאינטרפולציה ביליניארית - מציאת 4 השכנים
    x0 = np.floor(src_x).astype(int)
    y0 = np.floor(src_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # חסימת חריגות (Clipping) כדי למנוע קריסה מחוץ לגבולות המערך
    x0_clamped = np.clip(x0, 0, W - 1)
    x1_clamped = np.clip(x1, 0, W - 1)
    y0_clamped = np.clip(y0, 0, H - 1)
    y1_clamped = np.clip(y1, 0, H - 1)

    # חישוב משקלי האלפא והבטא (המרחק מהפיקסל הפינתי)
    alpha = (src_x - x0).reshape(H, W, 1)
    beta = (src_y - y0).reshape(H, W, 1)

    # 5. שליפת ערכי הפיקסלים של השכנים (דגימה וקטורית)
    I00 = image[y0_clamped, x0_clamped].reshape(H, W, C)
    I01 = image[y0_clamped, x1_clamped].reshape(H, W, C)
    I10 = image[y1_clamped, x0_clamped].reshape(H, W, C)
    I11 = image[y1_clamped, x1_clamped].reshape(H, W, C)

    # 6. חישוב הנוסחה הביליניארית בצורה וקטורית
    output = (1 - alpha) * (1 - beta) * I00 + \
             alpha * (1 - beta) * I01 + \
             (1 - alpha) * beta * I10 + \
             alpha * beta * I11

    # 7. טיפול באזורים שחורגים מגבולות התמונה (Masking)
    # פיקסל נחשב "תקני" רק אם כל 4 השכנים שלו היו בתוך התמונה המקורית
    valid_mask = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)
    valid_mask = valid_mask.reshape(H, W, 1)

    # איפוס פיקסלים חורגים (צביעת הרקע בשחור)
    output = np.where(valid_mask, output, 0)

    return output.astype(image.dtype)