import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# 加载 YOLO 模型
model = YOLO('yolov5su.pt')

# 加载手势分类模型
gesture_model = load_model('rock_paper_scissors_model.h5')

# 重新编译手势分类模型
gesture_model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# 定义手势类别
GESTURES = ["rock", "scissors", "paper"]

# 定义一个函数来分类手势
def classify_hand_shape(image):
    # 预处理图像
    image = cv2.resize(image, (150, 150))  # 调整图像大小以匹配模型输入
    image = image / 255.0  # 归一化
    image = np.expand_dims(image, axis=0)  # 添加批次维度

    # 使用模型进行预测
    predictions = gesture_model.predict(image)
    gesture_index = np.argmax(predictions)  # 获取预测结果的最大值索引
    return GESTURES[gesture_index]  # 返回对应的手势类别

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    # 遍历所有检测到的手
    for result in results:
        for box in result.boxes.xyxy:  # 遍历每个检测框
            x1, y1, x2, y2 = box.tolist()  # 获取检测框坐标
            hand_image = rgb_frame[int(y1):int(y2), int(x1):int(x2)]  # 裁剪手部图像

            # 如果裁剪的图像有效，则进行分类
            if hand_image.size > 0:
                hand_shape = classify_hand_shape(hand_image)  # 分类手势
                # 在图像上绘制检测框和手势类别
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, hand_shape, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('手势检测', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()