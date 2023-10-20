import cv2
import numpy as np
import tqdm
import math
import glob
import os
import pickle
import copy


CLASSES = ['bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve']
balance = False
synthetic_range = 1 if balance else 1
num_object_add = np.array([259,365,331,335,412,400,449,474,179,389])
#shadow_distances = [136, 107, 142, 131, 160, 111, 187, 188, 74, 98, 192]
#shadow_std = [19.88909036732824, 25.31810688817013, 27.279395864800293, 21.12536636193223, 20.385662264176116, 24.35794087290239, 18.680297550205932, 14.870493324665293, 24.161098339913597, 20.04714301638969]
#shadow_mean = [66.54238447816674, 103.4220909478466, 75.24109767141326, 70.69630847725782, 76.17708447058757, 99.97018854119202, 94.41155331278104, 99.7837049826508, 70.52279033988064, 57.12974663777573]
#shadow_ratio = [0.8565737321817516, 0.9465038174908947, 0.8081940232718029, 0.8003427428853545, 0.7287991836646665, 0.815980475245443, 0.8975803414791359, 0.943965701519078, 0.7114306252106289, 0.897584844623685]

# 2 Iter
shadow_distances = [76, 112, 0, 122, 143, 132, 0, 0, 90, 86, 0]
shadow_ratio = [0.879364120324713, 0.9440982056508929, 0, 0.8091671174464808, 0.7322822904833751, 0.8364885202646968, 0, 0, 0.7467756356814191, 0.896546440916789]
shadow_std = [21.186071897867496, 25.275333625382075, 0, 21.044879453332797, 20.270996348678466, 25.30292759817206, 0, 0, 27.512776331671326, 20.188694604422125]

RS = False
overlap = False
contrast = True
camera_pos = (570,160)
inpainting = ['', 'Pix']
object_probs = ['heatmap']

with open('heatmap_gaussian.pkl', 'rb') as f:
    heatmap = pickle.load(f)
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def distance(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    return np.linalg.norm(p2 - p1)

def get_angle(x1, y1, x2, y2):
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    v = p2 - p1
    return np.arctan2(v[1], v[0]) * 180 / np.pi

def find_starting_point(image, j):
    for i in range(image.shape[0]):
        if image[i][j] != 0:
            return i
    return None

def draw_line_with_angle(img, start_point, length, angle, color, thickness=1):
    """각도에 따라 선분 그리기"""
    end_point = (int(start_point[0] - length * math.cos(math.radians(angle))),
                 int(start_point[1] - length * math.sin(math.radians(angle))))
    cv2.line(img, start_point, end_point, color, thickness)

def generate_random_coordinates_2d(heatmap):
    # heatmap을 확률 분포로 정규화합니다.
    heatmap = heatmap / np.sum(heatmap)

    # heatmap의 크기를 가져옵니다.
    rows, cols = heatmap.shape
    
    # 랜덤 좌표를 선택합니다.
    indices = np.arange(rows * cols)
    random_index = np.random.choice(indices, p=heatmap.flatten())
    
    # 선택된 인덱스를 2차원 좌표로 변환합니다.
    random_row = random_index // cols
    random_col = random_index % cols
    
    return random_row, random_col

def create_gaussian_kernel(size_y, size_x, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - size_x//2)**2 + (y - size_y//2)**2) / (2 * sigma**2)),
        (size_x, size_y)
    )
    return kernel / np.sum(kernel)

def blur_edges(image, kernel_size=5, blur_amount=10):
    gray_scaled = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray_scaled, (kernel_size, kernel_size), 0)
    
    # Threshold the blurred image to create a binary image
    _, thresholded = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image
    result = np.copy(image)
    
    # Iterate over the contours
    for contour in contours:
        # Blur the region enclosed by each contour
        x, y, w, h = cv2.boundingRect(contour)
        result[y:y+h, x:x+w] = cv2.blur(result[y:y+h, x:x+w], (blur_amount, blur_amount))
    
    return result


if __name__ == '__main__':
    for synthetic_ratio in [synthetic_range]:   #range(1, synthetic_range+1):
        for object_prob in object_probs:
            np.random.seed(synthetic_ratio)
            folder = 'result/3000_' + object_prob
            '''
            if balance:
                folder += '_balance'
            else:
                folder += '_unbalance'
            '''
            if contrast:
                folder += '_contrast' 
            if overlap:
                folder += '_overlap' 
            for path in inpainting:
                    if RS:
                        path += '_rs'
                    createFolder(f'{folder}_{path}/Images')
                    createFolder(f'{folder}_{path}/Masks')

                    createFolder(f'{folder}_{path}_shadow/Images')
                    createFolder(f'{folder}_{path}_shadow/Masks')
            print(folder)
            object_group = dict()

            bg_group = glob.glob('./data/removed/*.png')
            

            object_group = {c: glob.glob(f'./data/object_Images/{i+1}/*.png') for i, c in enumerate(CLASSES)}
            object_img_group = {c: glob.glob(f'./data/object_Images/imgs/{i+1}/*.png') for i, c in enumerate(CLASSES)}

            crop_mask = np.zeros((480, 320), dtype=np.uint8)
            for hh in range(480):
                for ww in range(320):
                    if abs(math.atan2(ww-camera_pos[1], camera_pos[0]-hh)) <= math.radians(15) and 110 < math.dist([hh, ww],[camera_pos[0], camera_pos[1]]) < 540:
                        crop_mask[hh][ww] = 1

            iter_range = tqdm.tqdm(range(5000)) if (balance) else tqdm.tqdm(range(3000))
            for i in iter_range:
                background_name = np.random.choice(bg_group)[-8:]
                backgrounds = []
                backgrounds.append(cv2.imread(f'./data/removed/{background_name}', cv2.IMREAD_GRAYSCALE))
                for bg in inpainting[1:]:
                    backgrounds.append(cv2.imread(f'./{bg}/{background_name}', cv2.IMREAD_GRAYSCALE))

                shadow_backgrounds = copy.deepcopy(backgrounds)
                mask_background = np.zeros((480,320), dtype=np.uint8)
                test_background = np.zeros((480,320), dtype=np.uint8)

                with open(f'./data/removed/{background_name}'[:-3]+'pkl', 'rb') as f:
                    bg_info = pickle.load(f)
                if object_prob == 'bg':
                    objects_num = len(bg_info)
                else:
                    objects_num = np.random.randint(1,4) # 
                if num_object_add.sum() <= 0:
                        break
                if balance:
                    p = num_object_add / num_object_add.sum()
                    random_objects = np.random.choice(range(len(CLASSES)), objects_num, p = p, replace=False)
                    for r in random_objects:
                        num_object_add[r] -= 1
                else:
                    random_objects = np.random.choice(range(len(CLASSES)), objects_num, replace=False)

                obj_imgs = []
                obj_masks = []
                obj_x = []
                obj_y = []
                obj_contrast = []
                obj_mean = []

                for index, c in enumerate(random_objects):
                    object_name = CLASSES[c]
                    obj_select = np.random.choice(object_img_group[object_name])
                    obj_imgs.append(cv2.imread(obj_select, cv2.IMREAD_GRAYSCALE))
                    obj_masks.append(cv2.imread(f'./data/object_Images/masks/{c+1}/{obj_select[-8:-4]}_mask.png', cv2.IMREAD_GRAYSCALE))
                    with open(obj_select[:-3]+'pkl', 'rb') as f:
                        info = pickle.load(f)

                    if object_prob == 'bg':
                        obj_x.append(bg_info[index]['cx'])
                        obj_y.append(bg_info[index]['cy'])
                    elif object_prob == 'obj':
                        obj_x.append(info['cx'])
                        obj_y.append(info['cy'])
                    elif object_prob == 'random':
                        obj_x.append(np.random.randint(0,320))
                        obj_y.append(np.random.randint(0,480))
                    elif object_prob == 'heatmap':
                        cy, cx = generate_random_coordinates_2d(heatmap[object_name])
                        obj_x.append(cx)
                        obj_y.append(cy)
                    obj_contrast.append(info['contrast'])
                    obj_mean.append(info['mean'])
                    if RS:
                        M = cv2.getRotationMatrix2D((obj_imgs[-1].shape[0]//2, obj_imgs[-1].shape[1]//2),np.random.rand()*30-15, np.random.rand()*0.1+0.95) # -15~15, 0.95~1.05
                        obj_imgs[-1] = cv2.warpAffine(obj_imgs[-1], M,(obj_imgs[-1].shape[0], obj_imgs[-1].shape[1]))
                    
                    # 위치 찾기
                    std = 1
                    while True:
                        mask_not_ok = False
                        obj_mask_background = np.zeros((480,320), dtype=np.uint8)
                        obj_mask_test_background = np.zeros((480,320), dtype=np.uint8)
                        
                        offset_y = obj_imgs[-1].shape[0] // 2
                        offset_x = obj_imgs[-1].shape[1] // 2

                        y_start = max(0, obj_y[-1] - offset_y)
                        y_end = min(480, obj_y[-1] - offset_y + obj_masks[-1].shape[0])
                        x_start = max(0, obj_x[-1] - offset_x)
                        x_end = min(320, obj_x[-1] - offset_x + obj_masks[-1].shape[1])
                        obj_mask_test_background[y_start:y_end, x_start:x_end] = obj_imgs[-1][:y_end-y_start, :x_end-x_start]
                        mask_not_ok = np.bitwise_and(test_background[y_start:y_end, x_start:x_end],
                                                     obj_imgs[-1][:y_end-y_start, :x_end-x_start]).any()  # 겹침
                        if overlap:
                            mask_not_ok = False
                        
                        if not mask_not_ok:
                            offset_y = obj_masks[-1].shape[0] // 2
                            offset_x = obj_masks[-1].shape[1] // 2

                            y_start = max(0, obj_y[-1] - offset_y)
                            y_end = min(480, obj_y[-1] - offset_y + obj_masks[-1].shape[0])
                            x_start = max(0, obj_x[-1] - offset_x)
                            x_end = min(320, obj_x[-1] - offset_x + obj_masks[-1].shape[1])
                            # 겹치는 영역 제외하고 갱신
                            obj_mask_background[y_start:y_end, x_start:x_end] = obj_masks[-1][:y_end-y_start, :x_end-x_start]
                            mask_background[obj_mask_background > 0] = 0  # 겹치는 영역을 0으로 설정

                            gaussian_kernel = create_gaussian_kernel(obj_masks[-1].shape[0], obj_masks[-1].shape[1], obj_masks[-1].shape[0]/3)
                            gaussian_kernel = gaussian_kernel/gaussian_kernel.max()

                            #obj_masks[-1][gaussian_kernel < 0.5] = 0

                            mask_background[y_start:y_end, x_start:x_end] = obj_masks[-1][:y_end-y_start, :x_end-x_start] 
                            test_background += obj_mask_test_background
                            break
                        if object_prob == 'bg':
                            obj_x[-1] = np.clip(bg_info[index]['cx']+int(np.random.normal(0,std)), 0, 320)
                            obj_y[-1] = np.clip(bg_info[index]['cy']+int(np.random.normal(0,std)), 0, 480)
                        elif object_prob == 'obj':
                            obj_x[-1] = np.clip(info['cx']+int(np.random.normal(0,std)), 0, 320)
                            obj_y[-1] = np.clip(info['cy']+int(np.random.normal(0,std)), 0, 480)
                        elif object_prob == 'random':
                            obj_x[-1] = np.random.randint(0,320)
                            obj_y[-1] = np.random.randint(0,480)
                        elif object_prob == 'heatmap':
                            obj_x[-1] = np.clip(cx+int(np.random.normal(0,std)), 0, 320)
                            obj_y[-1] = np.clip(cy+int(np.random.normal(0,std)), 0, 480)
                        if std > 100:
                            np.delete(random_objects, -1, 0)
                            break
                        std += 1

                    if object_name not in ['shampoo-bottle','standing-bottle', 'valve', 'chain']:
                        shadow_mask = np.zeros((480,320), dtype=np.float64)
                        object_pos = [obj_y[-1], obj_x[-1]]
                        shadow_dist = shadow_distances[c]
                        
                        shadow_angle = get_angle(object_pos[1], object_pos[0], camera_pos[1], camera_pos[0])
                        for y in range(obj_masks[-1].shape[0]):
                            for x in range(obj_masks[-1].shape[1]):
                                if obj_masks[-1][y, x] > 0:
                                    target_y = obj_y[-1] - offset_y + y
                                    target_x = obj_x[-1] - offset_x + x
                                    if target_y < 480 and target_x < 320 and target_y >= 0 and target_x >= 0:
                                        draw_line_with_angle(shadow_mask, [target_x, target_y], shadow_dist, shadow_angle, 1.0, thickness=1)

                        shadow_random = np.clip(shadow_mask * np.random.normal(shadow_ratio[c], shadow_std[c]/255, shadow_backgrounds[0].shape), 0, 1)
                        shadow_random[shadow_random==0] = 1
                        for ii in range(len(shadow_backgrounds)):
                            shadow_backgrounds[ii] = (shadow_backgrounds[ii] * shadow_random).astype(np.uint8)

                for obj_id, c in enumerate(random_objects):
                    object_name = CLASSES[c]
                    offset_y = obj_imgs[obj_id].shape[0] // 2
                    offset_x = obj_imgs[obj_id].shape[1] // 2

                    y_start = max(0, obj_y[obj_id] - offset_y)
                    y_end = min(480, obj_y[obj_id] - offset_y + obj_imgs[obj_id].shape[0])
                    x_start = max(0, obj_x[obj_id] - offset_x)
                    x_end = min(320, obj_x[obj_id] - offset_x + obj_imgs[obj_id].shape[1])

                    for bb, bg in enumerate(backgrounds + shadow_backgrounds):
                        if contrast:
                            bg_obj = backgrounds[bb % len(backgrounds)][y_start:y_end, x_start:x_end][obj_imgs[obj_id][:y_end-y_start, :x_end-x_start] > 0]
                            non_zero_bg_obj = bg_obj[bg_obj!=0]

                            if len(non_zero_bg_obj) == 0:
                                insert_img = obj_imgs[obj_id]
                            else:
                                bg_mean = np.mean(non_zero_bg_obj)            # 오브젝트 평균
                                contrast_adjust = bg_mean-obj_mean[obj_id]
                                insert_img = np.clip((obj_imgs[obj_id]+contrast_adjust),0,255).astype(np.uint8)
                                if object_name in ['shampoo-bottle','standing-bottle']:                             # 2안
                                    insert_img = np.clip((obj_imgs[obj_id]/obj_mean[obj_id]*bg_mean),0,255).astype(np.uint8)

                        else:
                            insert_img = obj_imgs[obj_id]
                        
                        alpha = create_gaussian_kernel(obj_imgs[obj_id].shape[0], obj_imgs[obj_id].shape[1], (obj_imgs[obj_id].shape[0]/4))
                        alpha = alpha/alpha.max()
                        alpha[obj_imgs[obj_id] == 0] = 0
                        alpha[:y_end-y_start, :x_end-x_start][bg[y_start:y_end, x_start:x_end] == 0] = 1
                        alpha[:y_end-y_start, :x_end-x_start][mask_background[y_start:y_end, x_start:x_end] != 0] = 1

                        bg[y_start:y_end, x_start:x_end] = bg[y_start:y_end, x_start:x_end] * (1 - alpha)[:y_end-y_start, :x_end-x_start] + (insert_img * alpha)[:y_end-y_start, :x_end-x_start]

                mask_background *= crop_mask
                for j,path in enumerate(inpainting):
                    if RS:
                        path += '_rs'
                    backgrounds[j] *= crop_mask
                    shadow_backgrounds[j] *= crop_mask
                    cv2.imwrite(f'./{folder}_{path}/Masks/{str(i).zfill(4)}.png', mask_background)
                    cv2.imwrite(f'./{folder}_{path}/Images/{str(i).zfill(4)}.png', backgrounds[j])

                    cv2.imwrite(f'./{folder}_{path}_shadow/Masks/{str(i).zfill(4)}.png', mask_background)
                    cv2.imwrite(f'./{folder}_{path}_shadow/Images/{str(i).zfill(4)}.png', shadow_backgrounds[j])