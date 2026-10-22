深度学习yolov8如何训练 道路交通违规违章行为检测数据集 通过训练出的权重，来推理检测信号冲突，未佩戴安全帽，压线数据集、违规更改路线数据集，建立深度学习交通违规违章行为检测系统
@[toc]
下面只是为代码示例。作为参考学习。
交通行驶违规驾驶违章行为检测不规范行为情况数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/59463c15c8494206ba4bd5c863e4660f.png)

交通违章状态识别数据
        其中数据中的交通违章场景包括：信号冲突，未佩戴安全帽，压线、违规更改路线，四个大类场景进行展开。
        ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3bac3b4a43b747589c42f40ff7e73512.png)

     交通路拍数据，每四大类场景中按小类细分，比如信号冲突中包含（红灯时左转、红灯时直行、..），压线包含（压中心黄线、压双实线），违反更改路线中包含（违反改变一般道路路线、隧道路径更改冲突）

--
### 表格

| 类别         | 具体违规行为           | 采集比例 (Sheet) | 百分比 (%) |
|--------------|------------------------|------------------|------------|
| **信号冲突** | 红灯时左转             | 149,448          | 10.03      |
|              | 红灯直行               | 462,703          | 31.05      |
|              | 在左转向灯处直行       | 62,222           | 4.18       |
| **未佩戴安全帽** | 未佩戴安全帽           | 16,626           | 1.12       |
| **中心线冲突** | 中心线橙色实线冲突     | 186,380          | 12.51      |
|              | 中心线橙色双实线违规   | 263,352          | 17.67      |
| **违反更改路线** | 违反改变一般道路的路线 | 332,848          | 22.34      |
|              | 隧道隧道路径更改冲突   | 16,421           | 1.10       |
| **小计**     |                        | **1,490,000**    |            |

---


 ![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e699767dd3264ec488825f04aa5dd992.png)

使用 **YOLOv8** 来训练基于 **交通行驶不规范情况图像数据集** 的模型，我们需要经历一系列步骤，包括环境配置、数据预处理、模型训练、推理与部署等。以下是详细的指南和代码示例。
仅供参考学习。

### 一、环境准备

首先，确保安装了必要的库：

```bash
pip install ultralytics opencv-python numpy tqdm pycocotools
```

### 二、数据预处理

由于原始标签为 JSON 格式，需要将其转换为 YOLOv8 所需的标准格式（每张图片对应一个 `.txt` 文件，每行表示一个目标 `class_id x_center y_center width height`）。

#### 示例代码：JSON 转换为 YOLOv8 格式

```python
import os
import json
import cv2
from glob import glob
from tqdm import tqdm

# 设置路径
image_dir = 'data/images'
json_dir = 'data/jsons'
output_dir = 'data/yolo_labels'

os.makedirs(output_dir, exist_ok=True)

# 类别映射（根据你的具体子类修改）
class_mapping = {
    "red_left": 0,
    "red_straight": 1,
    "no_helmet": 2,
    "yellow_line": 3,
    "double_solid_line": 4,
    # 添加更多子类...
}

def convert_bbox(img_w, img_h, polygon):
    """将多边形坐标转换为最小外接矩形"""
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    xc = (xmin + xmax) / 2 / img_w
    yc = (ymin + ymax) / 2 / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return [xc, yc, w, h]

def process_json(json_path, image_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    label_path = os.path.join(output_dir, os.path.basename(json_path).replace('.json', '.txt'))
    with open(label_path, 'w') as f:
        for obj in data['objects']:
            cls_name = obj['class']
            if cls_name not in class_mapping:
                continue
            cls_id = class_mapping[cls_name]
            bbox = convert_bbox(w, h, obj['polygon'])
            line = f"{cls_id} {' '.join(map(str, bbox))}\n"
            f.write(line)

# 遍历所有 JSON 和图片文件
json_files = glob(os.path.join(json_dir, '*.json'))
for js in tqdm(json_files):
    img_path = os.path.join(image_dir, os.path.basename(js).replace('.json', '.jpg'))
    if os.path.exists(img_path):
        process_json(js, img_path)

print("✅ JSON 标注已成功转换为 YOLOv8 格式")
```

### 三、划分数据集

使用 `sklearn` 的 `train_test_split` 进行划分：

```python
from sklearn.model_selection import train_test_split
import shutil

image_paths = glob('data/images/*.jpg')
train_val, test = train_test_split(image_paths, test_size=0.1, random_state=42)
train, val = train_test_split(train_val, test_size=0.1, random_state=42)

def copy_files(files, dest):
    os.makedirs(os.path.join(dest, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'labels'), exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(dest, 'images', os.path.basename(f)))
        label_file = os.path.splitext(f)[0].replace('images', 'yolo_labels') + '.txt'
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(dest, 'labels', os.path.basename(label_file)))

copy_files(train, 'dataset/train')
copy_files(val, 'dataset/val')
copy_files(test, 'dataset/test')

print("✅ 数据集已成功划分为 train/val/test")
```

### 四、创建 `data.yaml` 文件

```yaml
train: dataset/train/images
val: dataset/val/images
test: dataset/test/images

nc: 6  # 子类总数（根据你的实际类别数量调整）
names: ['red_left', 'red_straight', 'no_helmet', 'yellow_line', 'double_solid_line', ...]  # 替换为你自己的类别名
```

## 比如说：这种

### 1. 类别映射与标签转换

首先，将表格中的类别信息转化为模型可以理解的格式。在YOLOv8中，每个类别需要有一个唯一的ID，并且标注文件（`.txt`）中的每一行应该包含类别ID、边界框中心点坐标以及宽度和高度，所有这些都归一化到0-1之间。

#### 示例代码：创建类别映射字典

```python
class_mapping = {
    "红灯时左转": 0,
    "红灯直行": 1,
    "在左转向灯处直行": 2,
    "未佩戴安全帽": 3,
    "中心线橙色实线冲突": 4,
    "中心线橙色双实线违规": 5,
    "违反改变一般道路的路线": 6,
    "隧道隧道路径更改冲突": 7
}
```

### 2. 数据集划分与准备

确保同学你的数据集被正确地划分为训练集、验证集和测试集。这一步骤已经在之前的指南中详细说明了。

### 3. 创建 `data.yaml` 文件

基于新的类别信息更新 `data.yaml` 文件：

```yaml
train: dataset/train/images
val: dataset/val/images
test: dataset/test/images

nc: 8  # 根据你的具体类别数量调整
names: ['红灯时左转', '红灯直行', '在左转向灯处直行', '未佩戴安全帽', '中心线橙色实线冲突', '中心线橙色双实线违规', '违反改变一般道路的路线', '隧道隧道路径更改冲突']
```

### 4. 训练模型

使用之前提到的训练代码进行模型训练。

#### 示例代码：设置加权损失函数

```python
from ultralytics import YOLO
import torch

# 加载模型
model = YOLO('yolov8s.pt')

# 设置权重（根据类别比例反向计算）
weights = [1 / 0.1003, 1 / 0.3105, 1 / 0.0418, 1 / 0.0112, 1 / 0.1251, 1 / 0.1767, 1 / 0.2234, 1 / 0.011]
weights = torch.tensor(weights, dtype=torch.float32)

# 自定义损失函数
def custom_loss(pred, target):
    loss = model.loss(pred, target)
    loss *= weights[target[:, 0].long()]
    return loss.mean()

# 训练模型
results = model.train(
    data='data.yaml',
    imgsz=640,
    epochs=100,
    batch=16,
    name='traffic_violation_train',
    device='0' if torch.cuda.is_available() else 'cpu',
    project='runs/traffic',
    custom_loss=custom_loss  # 使用自定义损失函数
)
```

### 5. 性能评估与优化

训练完成后，通过验证集和测试集评估模型性能，关注mAP、Precision、Recall等指标。如果发现某些类别的识别效果不佳，同学可以考虑增加该类别的样本量或进一步调整模型参数和训练策略。


### 五、模型训练

#### 使用 Python API 方式

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # 或 yolov8m.pt / yolov8l.pt

results = model.train(
    data='data.yaml',
    imgsz=640,
    epochs=100,
    batch=16,
    name='traffic_violation_train',
    device='0' if torch.cuda.is_available() else 'cpu',
    project='runs/traffic'
)
```

#### 命令行方式

```bash
yolo task=detect mode=train model=yolov8s.pt data=data.yaml \
     imgsz=640 epochs=100 batch=16 name=traffic_violation_train \
     device=0 project=runs/traffic
```

### 六、验证与测试

```bash
yolo task=detect mode=val model=runs/traffic/traffic_violation_train/weights/best.pt data=data.yaml
```

### 七、推理代码（图像/视频/摄像头）

#### 单图推理

```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/traffic/traffic_violation_train/weights/best.pt')
results = model('path/to/test.jpg')

for r in results:
    im_array = r.plot()
    cv2.imshow("Detection", im_array)
    cv2.waitKey(0)
```

#### 视频流实时检测

```python
def detect_video(source=0):  # source可以是摄像头索引或RTSP地址
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Live Detection", annotated_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

detect_video()
```

### 八、模型导出用于部署

```bash
# 导出为 ONNX
yolo export model=best.pt format=onnx

# 导出为 TensorRT（NVIDIA GPU）
yolo export model=best.pt format=engine device=0

# 导出为 OpenVINO（Intel 设备）
yolo export model=best.pt format=openvino
```

### 九、性能评估

训练过程中会自动生成日志和图表，位于 `runs/traffic/traffic_violation_train/` 目录下，包括：

- `results.csv`：记录 loss、mAP 等指标变化
- `train_batch*.jpg`：展示训练时的增强图像
- `confusion_matrix.png`：混淆矩阵
- `PR_curve.png`：Precision-Recall 曲线

你可以使用 TensorBoard 查看更详细的训练曲线：

```bash
tensorboard --logdir runs/traffic/traffic_violation_train
```

以上文字及代码仅供参考学习！
