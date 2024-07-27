import torch
import src.managers as managers
import src.model_components as model_components
import torch.nn as nn

def set_up_weights_folder(weights_folder_path: str):
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

    for scale in [44,54]:
        print(f"[INFO] Training Models for training scale, S: {scale}")
        train_transform = managers.TransformManager.get_train_transform(s_min=scale, s_max=scale, img_height=32, img_width=32)                                    
        train_dl = managers.DataManager.get_train_dl(train_transform, BATCH_SIZE, NUM_WORKERS)
        loss_fn = nn.CrossEntropyLoss()
        for name in model_components.Arch.ARCH.keys():
            print(f"Model: {name}")
            model = model_components.VGG(name)
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
            
            file_path = f"{MODELS_FOLDER}/scale_{scale}/{name}_scale_{scale}.pth"
            managers.ModelManager.save(model, file_path )
