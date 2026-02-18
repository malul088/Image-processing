import math
import numpy as np
import matplotlib.pyplot as plt

def degreesToRad(degress):
    return degress*math.pi/180

def retData(rad):
    sin_val = math.sin(rad)
    cos_val = math.cos(rad)
    return  sin_val, cos_val

def r_30():
    sin_val, cos_val=retData(degreesToRad(30))
    mat=[[cos_val,-sin_val],[sin_val, cos_val]]
    return mat

def sx_2():
    mat=[[2,0],[0,1]]
    return mat

angles = [1, 5, 10, 30, 45, 180, 90, 0]

for deg in angles:
    rad= degreesToRad(deg)
    sin_val, cos_val = retData(rad)
    print(f"{deg},{rad},{sin_val},{cos_val}")

mat1=r_30()
print("---------r_30-----------")
print(mat1)
print("------------------------")
print()

mat2=sx_2()
print("---------sx_2-----------")
print(mat2)
print("------------------------")
print()

mat1= np.array(mat1)
mat2= np.array(mat2)

rs = mat1 @ mat2
sr = mat2 @ mat1
print("---------rs-----------")
print(rs)
print("------------------------")
print()

print("---------sr-----------")
print(sr)
print("------------------------")
print()

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

R = np.array(r_30())
rotated_rect = R @ rect

plt.plot(rotated_rect[0], rotated_rect[1], 'r-', linewidth=2, label="Rotated 30°")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Rectangle rotation using r_30")
plt.show()

scailed_rect = mat2 @ rect

plt.plot(scailed_rect[0], scailed_rect[1], 'r-', linewidth=2, label="Scailed by 2")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Rectangle scailed by 2 ")
plt.show()

rsOn = rs @ rect

plt.plot(rsOn[0], rsOn[1], 'r-', linewidth=2, label="rs On")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("rs On ")
plt.show()

srOn = sr @ rect

plt.plot(srOn[0], srOn[1], 'r-', linewidth=2, label="sr On")
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("sr On")
plt.show()


plt.figure(figsize=(6,6))

plt.plot(rect[0], rect[1], 'b-', linewidth=2, label="Original")
plt.plot(rotated_rect[0], rotated_rect[1], 'r-', linewidth=2, label="Rotate 30°")
plt.plot(scailed_rect[0], scailed_rect[1], 'g-', linewidth=2, label="Scale X2")
plt.plot(rsOn[0], rsOn[1], 'm-', linewidth=2, label="RS")
plt.plot(srOn[0], srOn[1], 'c-', linewidth=2, label="SR")

plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.title("All rectangles")
plt.grid(True)
plt.show()