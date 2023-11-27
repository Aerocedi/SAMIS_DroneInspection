from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use the model

    results = model.train(data="config.yaml", epochs=1)  # train the model

