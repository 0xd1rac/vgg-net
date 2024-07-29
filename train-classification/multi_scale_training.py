import torch
import src.managers as managers
import src.model_components as model_components
import torch.nn as nn
import os

def load_scale_54_models(folder_path:str):
    return [managers.ModelManager.load(os.path.join(folder_path,name)) for name in os.listdir(folder_path)]

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
    MODELS_FOLDER = "weights-classification"
    LEARNING_RATE = 1e-2

    scale_min = 44
    scale_max = 64 
    
    scale_54_models = load_scale_54_models(f"{MODELS_FOLDER}/scale_54")
    train_transform = managers.TransformManager.get_train_transform_multi_scale(s_min=scale_min, s_max=scale_max, img_width=IMG_DIM, img_height=IMG_DIM)
    train_dl = managers.DataManager.get_train_dl(train_transform, BATCH_SIZE, NUM_WORKERS)
    loss_fn = nn.CrossEntropyLoss()

    for model in scale_54_models:
        print(f"[INFO] Training Model {model.name} for scales {scale_min} - {scale_max}")
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LEARNING_RATE,
                                    momentum=MOMENTUM,
                                    weight_decay=WEIGHT_DECAY
                                    )

        epoch_loss_lis, epoch_acc_lis = managers.ModelManager.train(model=model,
                                        train_dl=train_dl,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=DEVICE,
                                        num_epochs=NUM_EPOCHS
                                        )
        
        file_path = f"{MODELS_FOLDER}/multiscale/{model.name}_multiscale_{scale_min}-{scale_max}.pth"
        managers.ModelManager.save(model, file_path)
        



