import sys
import math

def gradient_z(x, y):
    return math.cos(x), math.cos(y)


def main():
    if len(sys.argv) < 3:
        print("Usage: python gradient_descent_z.py <x_start> <y_start>")
        return

    x = float(sys.argv[1])
    y = float(sys.argv[2])

    LEARNING_RATE = 2
    NUM_ITERATIONS = 1000

    z_start = math.sin(x) + math.sin(y)
    print(f"Starting point:  x={x:.4f},  y={y:.4f},  z={z_start:.4f}")

    for i in range(NUM_ITERATIONS):
        dz_dx, dz_dy = gradient_z(x, y)
        x = x - LEARNING_RATE * dz_dx
        y = y - LEARNING_RATE * dz_dy

    z_min = math.sin(x) + math.sin(y)
    print(f"Minimum found:   x={x:.4f},  y={y:.4f},  z={z_min:.4f}")


if __name__ == "__main__":
    main()

