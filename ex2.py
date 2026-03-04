import math
import numpy as np
import matplotlib.pyplot as plt

def translation_matrix(a,b):
    mat=[[1,0,a],
         [0,1,b],
         [0,0,1]]
    return mat


def rotation_matrix(theta):
    theta = math.radians(theta)
    sin_val = math.sin(theta)
    cos_val = math.cos(theta)
    mat = [[cos_val, -sin_val, 0],
           [sin_val, cos_val, 0],
           [0, 0, 1]]
    return mat

def scale_matrix(sx, sy=None):
    if sy is None:
        sy = sx
    mat = [[sx, 0, 0],
           [0, sy, 0],
           [0, 0, 1]]
    return mat

def bilinear_interpolation(alpha, beta, I00, I01, I10, I11):
    val = (1-alpha)*(1-beta)*I00 + alpha*(1-beta)*I01 + (1-alpha)*beta*I10 + alpha*beta*I11
    return val

def nearest_neighbor(alpha, beta, I00, I01, I10, I11):
    a_round = round(alpha)
    b_round = round(beta)

    if b_round == 0:
        if a_round == 0:
            return I00
        else:
            return I01
    else:
        if a_round == 0:
            return I10
        else:
            return I11

#1.e

T_nc= np.array(translation_matrix(-100,-200))
r30= np.array(rotation_matrix(30))
Tc= translation_matrix(100,200)
mat = Tc @ r30 @  T_nc
print(mat)

#2.a
w, h = 2, 1
x = [-w/2, w/2, w/2, -w/2, -w/2]  # סוגר את המלבן
y = [h/2, h/2, -h/2, -h/2, h/2]
rect = np.array([x, y])
plt.plot(x, y, 'b-', linewidth=2)  # קו כחול
plt.axhline(0, color='black')      # ציר X
plt.axvline(0, color='black')      # ציר Y
plt.gca().set_aspect('equal')      # פרופורציות נכונות
plt.title("Rectangle centered at origin")
plt.show()

#2.b

R = np.array(rotation_matrix(30))
rect =[x, y, [1,1,1,1,1]]
rotated_rect = R @ rect

plt.plot(rotated_rect[0], rotated_rect[1], 'r-', linewidth=2, label="Rotated 30°")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Rectangle rotation 30 degrees")
plt.show()

#2.c

R = np.array(rotation_matrix(45))
rect =[x, y, [1,1,1,1,1]]
rotated_rect = R @ rect
S = np.array(scale_matrix(2,1))
scaled_rect = S @ rotated_rect

plt.plot(scaled_rect[0], scaled_rect[1], 'r-', linewidth=2, label="Rotated 45° Scales 2(X)")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Rectangle rotation 45 degrees and scaled 2(X)")
plt.show()


#2.d

R = np.array(rotation_matrix(45))
S = np.array(scale_matrix(2,1))
rect =[x, y, [1,1,1,1,1]]
scaled_rect = S @ rect
rotated_rect = R @ scaled_rect

plt.plot(scaled_rect[0], scaled_rect[1], 'r-', linewidth=2, label="Scales 2(X) Rotated 45°")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Rectangle scaled 2(X) and rotation 45 degrees")
plt.show()