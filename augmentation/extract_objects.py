import glob
import tqdm
import cv2
import numpy as np
import os
import pickle

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def pad_image_to_square(img):
    height, width = img.shape
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    if np.abs(height-width) % 2 != 0:
        margin[0] += 1

    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]
    return np.pad(img, margin_list, mode='constant')

def erase_zero(matrix):
    # Count the number of rows and columns in the matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Find the rows and columns that are all zeros
    zero_rows = [i for i in range(rows) if all(val == 0 for val in matrix[i])]
    zero_cols = [j for j in range(cols) if all(matrix[i][j] == 0 for i in range(rows))]

    # Find the rows and columns that have at least one non-zero value
    non_zero_rows = [i for i in range(rows) if i not in zero_rows]
    non_zero_cols = [j for j in range(cols) if j not in zero_cols]

    # Extract the non-zero submatrix
    submatrix = np.array([[matrix[i][j] for j in non_zero_cols] for i in non_zero_rows])

    return submatrix

PATH = ''
CLASSES = ['background', 'bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve', 'wall']

classes_num = [0 for i in range(len(CLASSES))]

if __name__ == '__main__':
    for i in range(1, len(CLASSES)):
        createFolder(f'./{PATH}/object_Images/{i}')
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
            object = cv2.bitwise_and(img, img, mask=object_mask)
            label = np.unique(mask[contours[con][:, :, 1], contours[con][:, :, 0]])
            c = label[0]
            
            # 외곽선 경계상자를 구합니다.
            x, y, w, h = cv2.boundingRect(contours[con])
            # 경계상자의 중심점 계산
            cx = int(x + w / 2)
            cy = int(y + h / 2)

            c_img = pad_image_to_square(erase_zero(object))
            non_zero_obj = c_img[c_img!=0]
            obj_m = np.mean(non_zero_obj)            # 오브젝트 평균
            contrast = obj_m / bg_m
            # contrast, cx, cy
            obj_info = {'mean':obj_m,'contrast':contrast, 'cx':cx, 'cy':cy}
            bg_info.append({'mean':bg_m,'cx':cx, 'cy':cy})
            with open(f'./{PATH}/object_Images/{c}/{str(classes_num[c]).zfill(4)}.pkl', 'wb') as f:
                pickle.dump(obj_info, f)
            cv2.imwrite(f'./{PATH}/object_Images/{c}/{str(classes_num[c]).zfill(4)}.png', c_img)
            classes_num[c] = classes_num[c] + 1

        with open(f'./{PATH}/removed/{str(i).zfill(4)}.pkl', 'wb') as f:
                pickle.dump(bg_info, f)
        cv2.imwrite(f'./{PATH}/removed/{str(i).zfill(4)}.png', remove)