from ultralytics import YOLO

tfjs_model = YOLO("oysterTrainedModels/yolo11nOysterNick/weights/best_web_model", task="detect")

# Run inference
results = tfjs_model("https://ultralytics.com/images/bus.jpg")