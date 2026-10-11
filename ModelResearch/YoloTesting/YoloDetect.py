from ultralytics import YOLO

#Load trained model
model2 = YOLO("oysterTrainedModels/yolo11sMergedData960/weights/best.pt")

#run inference with model on an image
results = model2("15secoyster.mp4", save=True, show=True)