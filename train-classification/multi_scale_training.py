import torch
import src.managers as managers
import src.model as model
import torch.nn as nn

def load_scale_54_models(folder_paths:str):
    models = list()

def set_up_weights_folder(weights_folder_path:str):
    pass

if __name__ == "__main__":
    BATCH_SIZE = 256
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    DROPOUT_PROBA = 0.5
    # NUM_WORKERS = multiprocessing.cpu_count()
    NUM_WORKERS = 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_DIM = 32
    NUM_EPOCHS = 1
    MODELS_FOLDER = "weights"
    LEARNING_RATE = 1e-2

    scale_min = 44
    scale_max = 64 
    
    scale_54_models = load_scale_54_models("weights/scale_54")
    train_transform = managers.TransformManager.get_train_transform(scale_min, scale_max, IMG_DIM, IMG_DIM)
    train_dl = managers.DataManager.get_train_dl(train_transform, BATCH_SIZE, NUM_WORKERS)

    
    for model in scale_54_models:
        managers.ModelManager.train(model)
        file_path = ""
        managers.ModelManager.save(model)



