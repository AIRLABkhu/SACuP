import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RadioButtons
import tqdm

bg_group = glob.glob('./data/Images/*.png')
camera_pos = 700
CLASSES = ['background', 'bottle', 'can', 'chain',
           'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
           'standing-bottle', 'tire', 'valve', 'wall']

# opencv image BGR
color_map = {
    '0': [0, 0, 0], # black
    '1': [0, 0, 255], # red - bottle
    '2': [0, 125, 255], # orange - can
    '3': [0, 255, 255], # yellow - chain
    '4': [0, 255, 125], # spring green - drink-carton
    '5': [0, 255, 0], # green - hook
    '6': [255, 255, 0], # cyan - propeller
    '7': [255, 125, 0], # ocean - shampoo-bottle
    '8': [255, 0, 0], # blue - standing-bottle
    '9': [255, 0, 125], # violet - tire
    '10': [255, 0, 255], # magenta - valve
    '11': [125, 0, 255], # raspberry - wall
}

def colorization(imgs, gt):
    imgs = imgs.squeeze()
    imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)

    gt_color = np.zeros((gt.shape[0], gt.shape[1], 3)).astype(np.uint8)
    for i in range(0, len(color_map)):
        idx_gt = np.where(gt==int(i))
        gt_color[idx_gt] = color_map[f'{i}']

    img_pred = imgs * 0.7 + gt_color * 0.3
            
    return img_pred / 255

def pad_image(image):
    height, width = image.shape[:2]
    padded_height = camera_pos*2
    padded_width = camera_pos*2
    padded_image = np.zeros((padded_height, padded_width), dtype=image.dtype)
    padded_image[:480, camera_pos-160:camera_pos+160] = image
    return padded_image

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

line_len = 600
classes_line = [[] for i in range(len(CLASSES))]
#classes_line = [np.empty((0, line_len)) for i in range(len(CLASSES))] # [np.zeros(900) for i in range(len(CLASSES))]
classes_2d = [[] for i in range(len(CLASSES))] #
classes_len = [[] for i in range(len(CLASSES))]
classes_end = [[] for i in range(len(CLASSES))]

for bg in tqdm.tqdm(bg_group):
    bg_img = cv2.imread(bg, cv2.IMREAD_GRAYSCALE)
    bg_gt = cv2.imread('./230220_filter_full/Masks/'+bg[28:], cv2.IMREAD_GRAYSCALE)

    padded_img = pad_image(bg_img)
    padded_gt = pad_image(bg_gt)

    value = np.sqrt(((padded_img.shape[0]/2.0)**2.0)+((padded_img.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(padded_img, (camera_pos, camera_pos), value, cv2.WARP_FILL_OUTLIERS)
    polar_gt = cv2.linearPolar(padded_gt, (camera_pos, camera_pos), value, cv2.WARP_FILL_OUTLIERS)

    polar_image, ((x1, y1), (x2, y2)) = erase_zero(polar_image)
    polar_gt = polar_gt[y1:y2,x1:x2]

    polar_image = cv2.rotate(polar_image, cv2.ROTATE_90_CLOCKWISE)
    polar_gt = cv2.rotate(polar_gt, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow('a',polar_image)
    # cv2.waitKey()
    # 객체 간 경계선 추출
    contours, hierarchy = cv2.findContours(polar_gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for con in range(len(contours)):
        # i번째 객체 추출
        object_mask = np.zeros_like(polar_gt)
        cv2.drawContours(object_mask, contours, con, 255, -1)
        label = np.unique(polar_gt[contours[con][:, :, 1], contours[con][:, :, 0]])
        c = label[0]
        #print(c)
        x, y, w, h = cv2.boundingRect(contours[con])
        if h < 10:
            continue

        classes_len[c].append(polar_image.shape[0]-y)
        classes_end[c].append(h)

        for i in range(x,x+w):
            line_sum = np.zeros(line_len)
            line_sum[:polar_image.shape[0]-y] = polar_image[y:,i]
            line_sum = line_sum.reshape(1, line_len)
            classes_line[c].append(line_sum)
            # 반복문이 끝난 후에 한 번에 결합

        object_2d = np.zeros((line_len,480))
        object_2d[:polar_image.shape[0]-y, 240-w//2-1:240-w//2-1+w] = polar_image[y:,x:x+w]
        classes_2d[c].append(object_2d)

classes_line_group = [0]
classes_std_group = [0]
classes_2d_group = [0]
for i in tqdm.tqdm(range(1, len(CLASSES))):
    classes_line_np = np.array(classes_line[i]).squeeze()
    classes_line_mean = np.zeros(line_len, dtype=np.float64)
    classes_line_std = np.zeros(line_len, dtype=np.float64)
    for j in range(line_len):
        if (classes_line_np[:,j][classes_line_np[:,j]!=0]).size > 1:
            classes_line_mean[j] = classes_line_np[:,j][classes_line_np[:,j]!=0].mean()
            classes_line_std[j] = classes_line_np[:,j][classes_line_np[:,j]!=0].std()
        else:
            classes_line_mean[j] = 0
            classes_line_std[j] = 0

    classes_2d_np = np.array(classes_2d[i]).squeeze()
    classes_2d_mean = np.zeros((line_len,480))

    for j in range(line_len):
        for k in range(480):
            if (classes_2d_np[:,j,k][classes_2d_np[:,j,k]!=0]).size > 1:
                classes_2d_mean[j,k] = classes_2d_np[:,j,k][classes_2d_np[:,j,k]!=0].mean()
            else:
                classes_2d_mean[j,k] = 0
    classes_2d[i], ((x1, y1), (x2, y2)) = erase_zero(classes_2d_mean)
    img = cv2.cvtColor((classes_2d[i]/classes_2d[i].max()*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#    img[sum(classes_end[i])// len(classes_end[i]),:] = [0,0,255]
    classes_line_group.append(classes_line_mean)
    classes_std_group.append(classes_line_std)
    classes_2d_group.append(img)
    cv2.imwrite(f'./shadow_profile/{CLASSES[i]}_2d.png', img)

i=1
# 슬라이더 이벤트 핸들러
def update(val):
    col_index = int(slider.val)  # 슬라이더 값 가져오기

    # 그래프 업데이트
    vertical_line.set_xdata(col_index)
    horizontal_line.set_ydata(col_index)
    fig.canvas.draw_idle()  # 그래프 다시 그리기

def click(val):
    global std_line
    i = CLASSES.index(val)
    line1.set_ydata(classes_line_group[i])

    #slider.valmax = classes_2d_group[i].shape[0] - 1
    ax1.collections.remove(std_line)  # 이전에 그렸던 std_line 제거
    std_line = ax1.fill_between(range(classes_line_group[i].shape[0]), classes_line_group[i] - classes_std_group[i], classes_line_group[i] + classes_std_group[i], color='#888888', alpha=0.4)

    vertical_line.set_xdata(classes_2d_group[i].shape[1] // 2)
    horizontal_line.set_ydata(classes_2d_group[i].shape[1] // 2)
    ax3.imshow(classes_2d_group[i])
    fig.canvas.draw_idle()  # 그래프 다시 그리기

fig, (ax1, ax3) = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.25)  # 그래프 하단 여백 조절

# 슬라이더 위치 및 크기 설정
slider_ax = plt.axes([0.15, 0.1, 0.7, 0.03])
slider = Slider(slider_ax, '', 0, 600, valinit=0, valstep=1)
rax = plt.axes([0.8, 0.4, 0.2, 0.5])
radio_button = RadioButtons(rax, (['bottle', 'can', 'chain',
           'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
           'standing-bottle', 'tire', 'valve', 'wall']))

image_plot = ax3.imshow(classes_2d_group[i])
horizontal_line = ax3.axhline(classes_2d_group[i].shape[1] // 2, color='red', alpha=0.5)
ax3.invert_yaxis()

line1, = ax1.plot(classes_line_group[i])
vertical_line = ax1.axvline(classes_2d_group[i].shape[1] // 2, color='red', alpha=0.5)

ax1.axvline(sum(classes_end[i]) // len(classes_end[i]), color='blue', alpha=0.5) # 오브젝트가 끝나는 지점 평균
ax1.set_title(CLASSES[i])
ax1.set_ylim([0, 255])

std_line = ax1.fill_between(range(classes_line_group[i].shape[0]), classes_line_group[i] - classes_std_group[i], classes_line_group[i] + classes_std_group[i], color='#888888', alpha=0.4)
#plt.savefig(f'./shadow_profile/{CLASSES[i]}.png')

slider.on_changed(update)  # 슬라이더 이벤트 핸들러 연결
radio_button.on_clicked(click)
plt.show()


