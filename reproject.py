import cv2, numpy as np, math, os

############# FILL IN PARAMETERS HERE ###################
input_path = "video360.mp4"         # or .png/.jpg
output_path = "video360_repr.mp4"  # or .png
face_size = 1024                    # desired width and height of the output video
fov = 90 * math.pi / 180            # FOV in radians
#########################################################

def process_frame(img):
    h, w = img.shape[:2]
    u = np.linspace(-1, 1, face_size)
    v = np.linspace(-1, 1, face_size)
    u, v = np.meshgrid(u, v)
    f = 1 / math.tan(fov / 2)

    x, y, z = u, -v, np.ones_like(u) * f
    norm = np.sqrt(x*x + y*y + z*z)
    x, y, z = x/norm, y/norm, z/norm

    theta = np.arctan2(x, z)
    phi = np.arcsin(y)

    u_eq = (theta / (2 * math.pi) + 0.5) * w
    v_eq = (0.5 - phi / math.pi) * h

    map_x = u_eq.astype(np.float32)
    map_y = v_eq.astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

if input_path.lower().endswith((".png", ".jpg", ".jpeg")):
    img = cv2.imread(input_path)
    face = process_frame(img)
    cv2.imwrite(output_path, face)
else:
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (face_size, face_size))
    frame_nr = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        out.write(process_frame(frame))
        frame_nr += 1

    cap.release()
    out.release()

print("Done:", output_path)
