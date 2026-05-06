# validate.py - Test model accuracy (Week 6 evaluation)
from ultralytics import YOLO

def validate_model():
    model = YOLO('runs/detect/shelf_inventory/weights/best.pt')
    
    # Run validation on test set
    metrics = model.val(
        data='data/dataset.yaml',
        imgsz=640,
        batch=8,
        plots=True,  #generates confusion matrix, PR curves
        save_json=True
    )
    
    print(f"\n  Validation Results:")
    print(f"mAP@50: {metrics.box.map50:.3f}")
    print(f"mAP@50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    
    return metrics

if __name__ == "__main__":
    validate_model()