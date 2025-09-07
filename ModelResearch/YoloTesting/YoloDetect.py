from ultralytics import YOLO

#Load trained model
model2 = YOLO("oysterTrainedModels/yolo11sOysterNick/weights/best.pt")

#run inference with model on an image
results = model2("Untitled-video-Made-with-Clipchamp_mp4-0008_jpg.rf.a34c142a3dcd7dfc463507d628afb901.jpg", save=True, show=True)