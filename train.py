import src.managers as managers
import argparse
import json
import torch
import torch.nn as nn 
import src as src
import src.model_components

def load_config(config_path):
    """Load JSON config file."""
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

def run_single_scale_training(models,config):
    print(f"[INFO] Training models for training scale, S: {config['training_scale']}")

    train_transform = managers.TransformManager.get_test_transform_single_scale(
                                                scale=config['training_scale'], 
                                                img_width=config['img_width'],
                                                img_height=config['img_height']
                                                )
    train_dl = managers.DataManager.get_train_dl(train_transform,
                                                config['batch_size'],
                                                config['num_workers']
                                                )
    loss_fn = nn.CrossEntropyLoss()

    for model in models:
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=config['lr'],
                                momentum=config['momentum'],
                                weight_decay=config['weight_decay']
                                )
        
        epoch_loss_lis, epoch_acc_lis = managers.ModelManager.train(model=model,
                            train_dl=train_dl,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            device=config['device'],
                            num_epochs=config['num_epochs']
                            )
        
        file_path = f"{config['weights_folder_path']}/{model.name}/epoch_{len(model.epoch_acc_lis)}.pth"
        managers.ModelManager.save(model, file_path)

def run_multi_scale_training(models, config):
    print(f"[INFO] Training models for multiple training scales, S: [{config['s_min']}, {config['s_max']}]")
    train_transform = managers.TransformManager.get_train_transform_multi_scale(s_min=config['s_min'], 
                                                                                s_max=config['s_max'], 
                                                                                img_width=config['img_width'],
                                                                                img_height=config['img_height']
                                                                                )
    train_dl = managers.DataManager.get_train_dl(train_transform, 
                                                 config['batch_size'],
                                                 config['num_workers']
                                                 )

    loss_fn = nn.CrossEntropyLoss()

    for model in models:
        print(f"[INFO] Training Model {model.name} for scales {config['s_min']} - {config['s_max']}")
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['lr'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay']
                                    )
        
        epoch_loss_lis, epoch_acc_lis = managers.ModelManager.train(model=model,
                                        train_dl=train_dl,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        device=config['device'],
                                        num_epochs=config['num_epochs']
                                        )
        file_path = f"{config['weights_folder_path']}/{model.name}/epoch_{len(model.epoch_acc_lis)}.pth"
        managers.ModelManager.save(model, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to load and use a JSON config file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the JSON config file")
    parser.add_argument('--train-mode', type=str, choices=['multi-scale', 'single-scale'], required=True, help="Test mode options: multi-scale, single-scale")
    args = parser.parse_args()
    config = load_config(args.config)
    train_mode = args.train_mode
    print(f"Train Mode: {train_mode}")

    if config['preloads']:
        models = managers.ModelManager.load_models(config['preloads_weights_folder'])
    else:
        models = [src.model_components.VGG(name) for name in src.model_components.Arch.ARCH.keys()]

    
    if train_mode == 'single-scale':
        run_single_scale_training(models, config)
    elif train_mode == 'multi-scale':
        run_multi_scale_training(models, config)
    




    