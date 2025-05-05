import os

class Config:
    def __init__(self, 
                 dataset_path="dataset/", 
                 checkpoint_dir="outputs/checkpoints/",
                 epochs=15,
                 batch_size=32,
                 learning_rate=1e-4,
                 alpha=1.0,
                 beta=2.0):
        
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        
        self.DATASET_PATH = dataset_path
        self.CHECKPOINT_DIR = checkpoint_dir
        
    
    @property
    def IMAGES_PATH(self):
        return os.path.join(self.DATASET_PATH, "images")
    
    @property
    def ANNOTATIONS_PATH(self):
        return os.path.join(self.DATASET_PATH, "annotations", "coco.json")
    
    @property
    def FILTERED_ANNOTATIONS_PATH(self):
        return os.path.join(self.DATASET_PATH, "annotations", "filtered_annotations.json")
    

