import glob
import tqdm
import cv2
import numpy as np
import os
import pickle
import math
import copy

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def pad_image_to_square(img, mask):
    height, width = img.shape
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    if np.abs(height-width) % 2 != 0:
        margin[0] += 1

    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]
    return np.pad(img, margin_list, mode='constant'), np.pad(mask, margin_list, mode='constant')

def erase_zero(matrix):
    # Convert the matrix to NumPy array
    matrix = np.array(matrix)

    # Find the rows and columns that are all zeros
    zero_rows = np.all(matrix == 0, axis=1)
    zero_cols = np.all(matrix == 0, axis=0)

    # Find the rows and columns that have at least one non-zero value
    non_zero_rows = ~zero_rows
    non_zero_cols = ~zero_cols

    # Extract the non-zero submatrix
    submatrix = matrix[non_zero_rows][:, non_zero_cols]

    # Find the bounding rectangle coordinates
    contours, _ = cv2.findContours(matrix.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, w, h = cv2.boundingRect(contours[0])
        bounding_rect = [(x, y), (x + w, y + h)]
    else:
        bounding_rect = []

    return submatrix, bounding_rect

PATH = ''
CLASSES = ['background', 'bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve', 'wall']

classes_num = [0 for i in range(len(CLASSES))]

if __name__ == '__main__':
    for i in range(1, len(CLASSES)):
        createFolder(f'./{PATH}/object_Images/imgs/{i}')
        createFolder(f'./{PATH}/object_Images/masks/{i}')
    createFolder(f'./{PATH}/removed')
    imgs = glob.glob(f'./{PATH}/Images/*.png')
    masks = glob.glob(f'./{PATH}/Masks/*.png')
    for i in tqdm.tqdm(range(len(masks))):
        mask = cv2.imread(masks[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
        remove = img.copy()
        remove[mask > 0] = 0
        non_zero_background = remove[remove!=0]
        bg_m = np.mean(non_zero_background)            # 배경 평균
        bg_info = []
        # 객체 간 경계선 추출
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for con in range(len(contours)):
            # i번째 객체 추출
            object_mask = np.zeros_like(mask)
            cv2.drawContours(object_mask, contours, con, 255, -1)
            x, y, w, h = cv2.boundingRect(contours[con])
            # 외곽선 경계상자를 구합니다.
            # 경계상자의 중심점 계산
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            r = max(w, h)

            r = int(math.sqrt(math.pow(r,2)+math.pow(r,2))*2)
            label = np.unique(mask[contours[con][:, :, 1], contours[con][:, :, 0]])
            c = label[0]
            if c == 3:
                r = int(math.sqrt(math.pow(r,2)+math.pow(r,2)))
          

            kernel = np.ones((3, 3), np.uint8)
            object_mask = cv2.dilate(object_mask, kernel, iterations=20)

            object = cv2.bitwise_and(img, img, mask=object_mask)
            mask_crop = cv2.bitwise_and(mask, mask, mask=object_mask)

            #c_img = pad_image_to_square(erase_zero(object))
            c_img, ((x1, y1), (x2, y2)) = erase_zero(object)
           
            c_mask = mask_crop[y1:y2,x1:x2]
            c_img, c_mask = pad_image_to_square(c_img, c_mask)
            obj_bg = copy.deepcopy(c_img)
            obj_bg[c_mask>0] = 0
            non_zero_obj = obj_bg[obj_bg!=0]
            obj_m = np.mean(non_zero_obj)            # 오브젝트 평균

            contrast = obj_m / bg_m
            # contrast, cx, cy
            obj_info = {'mean':obj_m,'contrast':contrast, 'cx':cx, 'cy':cy}
            bg_info.append({'mean':bg_m,'cx':cx, 'cy':cy})
            with open(f'./{PATH}/object_Images3/imgs/{c}/{str(classes_num[c]).zfill(4)}.pkl', 'wb') as f:
                pickle.dump(obj_info, f)
            cv2.imwrite(f'./{PATH}/object_Images3/imgs/{c}/{str(classes_num[c]).zfill(4)}.png', c_img)
            cv2.imwrite(f'./{PATH}/object_Images3/masks/{c}/{str(classes_num[c]).zfill(4)}_mask.png', c_mask)
            
            classes_num[c] = classes_num[c] + 1

        with open(f'./{PATH}/removed/{str(i).zfill(4)}.pkl', 'wb') as f:
                pickle.dump(bg_info, f)
        cv2.imwrite(f'./{PATH}/removed/{str(i).zfill(4)}.png', remove)
