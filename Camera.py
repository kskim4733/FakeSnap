import cv2
import torch
from model import Net
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.autograd import Variable


def get_camera_frame(camera):
    ret, frame = camera.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return frame


def add_moustache(keypoints, face_img):
    keypoints = keypoints.data.numpy()
    keypoints = (keypoints * 50.0) + 100
    keypoints = np.reshape(keypoints, (68, -1))

    m_point_x = int(keypoints[51][0])
    m_point_y = int(keypoints[51][1]) - 20

    eye_left_coord_x = int(keypoints[36][0])

    eye_right_coord_x = int(keypoints[45][0])

    moustache_img = cv2.imread("stickers/stache.png", -1)
    moustache_width = int((eye_right_coord_x - eye_left_coord_x))
    offset = int(moustache_width // 2)
    moustache_height = int(moustache_width * (moustache_img.shape[0] / moustache_img.shape[1]))
    resized_moustache = cv2.resize(moustache_img, (moustache_width, moustache_height))

    alpha_s = resized_moustache[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        face_img[m_point_y : m_point_y + moustache_height, m_point_x - offset : m_point_x - offset + moustache_width, c] = \
            (alpha_s * resized_moustache[:, :, c] + alpha_l * face_img[m_point_y : m_point_y + moustache_height, m_point_x - offset : m_point_x - offset + moustache_width, c])

    return face_img

def add_sunglasses(keypoints, face_img):
    keypoints = keypoints.data.numpy()
    keypoints = (keypoints * 50.0) + 100
    keypoints = np.reshape(keypoints, (68, -1))
    
    eye_left_coord_x = int(keypoints[0][0])
    eye_left_coord_y = int(keypoints[0][1]) - 20
    
    eye_right_coord_x = int(keypoints[16][0])
    eye_right_coord_y = int(keypoints[16][1])

    sunglasses_img = cv2.imread("stickers/sg1.png", -1)
    sunglasses_width = int(eye_right_coord_x - eye_left_coord_x)
    sunglasses_height = int(sunglasses_width * (sunglasses_img.shape[0] / sunglasses_img.shape[1]))
    resized_sunglasses = cv2.resize(sunglasses_img, (sunglasses_width, sunglasses_height))

    alpha_s = resized_sunglasses[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        face_img[eye_left_coord_y : eye_left_coord_y + sunglasses_height, eye_left_coord_x: eye_left_coord_x + sunglasses_width, c] = \
            (alpha_s * resized_sunglasses[:, :, c] + alpha_l * face_img[eye_left_coord_y: eye_left_coord_y + sunglasses_height, eye_left_coord_x: eye_left_coord_x + sunglasses_width, c])

    return face_img


def visualize_points(image, keypoints, simplify=False): # this code visualizes key points on top of the camera frame
    keypoints = keypoints.data.numpy()
    keypoints = (keypoints * 50.0) + 100
    keypoints = np.reshape(keypoints, (68, -1))
    for i, (x, y) in enumerate(keypoints):
        x = int(x)
        y = int(y)
        # total 68 points
        if (simplify):
            # 0 is left 16 is right eye ends,
            if i == 0 or i == 16 or i == 36 or i == 45 or i == 51:
                image = cv2.circle(image, (x, y), 3, (0, 0, 255), thickness=2)
        else:
            if i < 36:
                image = cv2.circle(image, (x, y), 3, (255,0,0), thickness=2)  # BGR
                if i == 0 or i == 16: # end of jaw line
                    image = cv2.circle(image, (x, y), 3, (0, 0, 255), thickness=2)

            else:
                if i == 36 or i == 45: # two end of eye points
                    image = cv2.circle(image, (x, y), 3, (0, 0, 255), thickness=2)
                else:
                    image = cv2.circle(image, (x, y), 3, (0,255,0), thickness=2)
                    if i == 51: # underneath nose
                        image = cv2.circle(image, (x, y), 3, (0, 0, 255), thickness=2)
                    if i == 68 or i == 60:
                        image = cv2.circle(image, (x, y), 3, (0, 0, 255), thickness=2)


    return image


def run_model(model, input_image, device):
    input_image_tensor = np.expand_dims(input_image, 0)
    input_image_tensor = np.expand_dims(input_image_tensor, 0)
    input_image_tensor = Variable(torch.from_numpy(input_image_tensor))
    input_image_tensor = input_image_tensor.float().to(device)

    prediction = model(input_image_tensor).cpu().detach()
    return prediction


def transform_input(image):
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_image = cv2.resize(input_image, (224, 224))
    input_image = input_image / 255.0
    return input_image


def resize_image(image):
    input_image = cv2.resize(image, (224, 224))
    return input_image


def find_face(frame):
    face_cascade = cv2.CascadeClassifier('detector/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, 1.2, 2)
    return faces


def key_pts_checker(new_keypoints, old_keypoints):
    if old_keypoints is not None:
        new_keypoints = new_keypoints.data.numpy()
        new_keypoints = (new_keypoints * 50.0) + 100
        new_keypoints = np.reshape(new_keypoints, (68, -1))

        old_keypoints = old_keypoints.data.numpy()
        old_keypoints = (old_keypoints * 50.0) + 100
        old_keypoints = np.reshape(old_keypoints, (68, -1))

        new_point_x = int(new_keypoints[51][0])
        new_point_y = int(new_keypoints[51][1])

        old_point_x = int(old_keypoints[51][0])
        old_point_y = int(old_keypoints[51][1])
        a = np.asarray([new_point_x, new_point_y])
        b = np.asarray([old_point_x, old_point_y])
        return (np.linalg.norm(a - b) > 20)
    return True


def run(model_path):
    model = Net()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    camera = cv2.VideoCapture(cv2.CAP_DSHOW)
#  Init-----------------------------------------------------------------------------------------------------------------
    if not camera.isOpened():
        raise IOError("Cannot open WebCam")
    prev_keypts = None
    while True:
        try:
            frame = get_camera_frame(camera)
            faces = find_face(frame)
            if len(faces) > 0:  # returns x y w h
                for i in range(0, len(faces)):
                    face_coord = faces[i]
                    x, y, w, h = face_coord[0], face_coord[1], face_coord[2],  face_coord[3]
                    face = frame[y - 25: y + h + 25, x - 25: x + w + 25]
                    face_tf = transform_input(face)

                    key_points = run_model(model, face_tf, device)
                    if not key_pts_checker(key_points, prev_keypts):
                        key_points = prev_keypts
                    # temp = visualize_points(resize_image(face), key_points)
                    temp = add_moustache(key_points, resize_image(face))
                    temp = add_sunglasses(key_points, resize_image(temp))
                    face_patch = cv2.resize(temp, (face.shape[1], face.shape[0]))
                    # print("face shape: ", face.shape, "face_tf shape: ", face_tf.shape,  "face_patch shape: ", face_patch.shape)
                    frame[y - 25: y + h + 25, x - 25: x + w + 25] = face_patch
                    prev_keypts = key_points
            cv2.imshow('Input', frame)
            c = cv2.waitKey(5)
            if c == 20:
                break
        except:
            continue





if __name__ == "__main__":
    #norm_flip_500
    # norm_100_old.pt
    #norm_flip.pt
    # run("C:\\Users\\kskim\\Desktop\\train-test-data\\gold.pt")
    run("models/gold.pt")