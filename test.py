# spell-checker: disable
import dlib
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the model
model = load_model('face_model.h5')

# Load the face detector
detector = dlib.get_frontal_face_detector()

# open the camera
cap = cv2.VideoCapture(0)

# lable 0 is Ben, 1 is Paddy
ResultMap = {0: 'Ben', 1: 'Paddy'}

while True:
    # 读取帧
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 使用dlib检测面部
    faces = detector(gray)
    
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # 提取面部区域并进行预处理
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))  
        face_img = face_img / 255.0  # 归一化
        face_img = np.expand_dims(face_img, axis=0)  # 增加批次维度
        
        # 使用模型进行预测
        prediction = model.predict(face_img)
        predicted_label = ResultMap[np.argmax(prediction)]
        
        # 在面部周围绘制矩形并显示预测结果
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Label: {predicted_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    # 显示结果帧
    cv2.imshow('Frame', frame)
    
    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()