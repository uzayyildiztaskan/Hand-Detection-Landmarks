class Config:
    def __init__(self, 
                 dataset_path="dataset/FreiHAND_pub_v2", 
                 checkpoint_dir="outputs/checkpoints/",
                 epochs=15,
                 batch_size=32,
                 learning_rate=1e-4,
                 alpha=1.0,
                 beta=2.0):
        
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.ALPHA = alpha
        self.BETA = beta
        
        self.DATASET_PATH = dataset_path
        self.CHECKPOINT_DIR = checkpoint_dir
        
    
    @property
    def RGB_FOLDER_PATH(self):
        return f"{self.DATASET_PATH}/training/rgb"
    
    @property
    def KEYPOINT_ANNOTATION_3D_PATH(self):
        return f"{self.DATASET_PATH}/training_xyz.json"
    
    @property
    def INTRINSIC_CAMERA_MATRIX_PATH(self):
        return f"{self.DATASET_PATH}/training_K.json"
    
    @property
    def KEYPOINT_ANNOTATION_2D_PATH(self):
        return f"{self.DATASET_PATH}/training_kp2d.npy"

