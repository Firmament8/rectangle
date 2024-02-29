from ultralytics import YOLO

model = YOLO('rectangle.pt')
model.predict(source='./images', conf=0.5, save=True, name='output')
#检测图像路劲在.\images下 结果保存在 .\runs\detect\ 下