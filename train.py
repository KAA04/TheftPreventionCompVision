from ultralytics import YOLO

def train_model():
    #load pretrained weights as starting point
    model = YOLO('yolov8n.pt')
    
    #train on our dataset
    results = model.train(
        data='dataset\Data\data.yaml',  #GAP: will change after data is provided
        epochs=100,
        imgsz=640,   #512 or 416 if there are memory issues
        batch=8,
        device=0,
        project='runs\detect',
        name='shelf_inventory',
        patience=15,    #early stopping if no improvement
        save=True,
        plots=True,    #save training curves
        val=True    #run validation during training
    )
    
    print("\n---Training complete---")
    print(f"Best model saved to: runs/detect/shelf_inventory/weights/best.pt")
    
    return results

if __name__ == "__main__":
    train_model()