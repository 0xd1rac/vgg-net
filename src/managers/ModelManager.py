from src.common_imports import *
from tqdm import tqdm
import src.model_components as model_components
import os
import numpy as np

class ModelManager():
    @staticmethod
    def train_one_epoch(model: model_components.VGG,
                        train_dl: DataLoader,
                        loss_fn: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        device = torch.device
                        ) -> Tuple[float, float]:
        batch_loss_lis = []
        crt_preds = 0
        total_preds = 0

        for batch in tqdm(train_dl, desc="Training", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            batch_loss = loss_fn(outputs, labels)
            batch_loss.backward() # backward pass
            optimizer.step() # update the parameters

            batch_loss_lis.append(batch_loss.item())
            _, predicted = outputs.max(1)

            crt_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
        
        epoch_loss = sum(batch_loss_lis) / len(batch_loss_lis)
        epoch_acc = crt_preds / total_preds

        return epoch_loss, epoch_acc


    @staticmethod
    def train(model: model_components.VGG,
              train_dl: DataLoader, 
              loss_fn: nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              num_epochs: int
              ) -> Tuple[List[float], List[float]]:
        

        model.to(device)
        model.train()

        # Contains the losses and accuracies for each epoch
        epoch_loss_lis = []
        epoch_acc_lis = []

        for _ in range(num_epochs):
            epoch_loss, epoch_acc = ModelManager.train_one_epoch(model,train_dl,loss_fn,optimizer,device)
            epoch_loss_lis.append(epoch_loss)
            epoch_acc_lis.append(epoch_acc)
        
        model.epoch_loss_lis.extend(epoch_loss_lis)
        model.epoch_acc_lis.extend(epoch_acc_lis)

        return epoch_loss_lis, epoch_acc_lis

    @staticmethod
    def predict(model: model_components.VGG, 
                dataloader: DataLoader, 
                device: torch.device
                ):
        model.to(device)
        model.eval()
        all_preds, all_labels, all_probs = [] , [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference", leave=False):
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                
                _, batch_preds = outputs.max(1)
                
                all_probs.append(outputs.cpu().numpy())
                all_preds.append(batch_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        return torch.tensor(all_preds), torch.tensor(all_labels),torch.tensor(all_probs)


    @staticmethod
    def save(model: model_components.VGG, 
             file_path: str
            ):
        assert file_path != None, "Model file path not defined"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)


        torch.save(
            {
                'model_name': model.name,
                'model_state_dict': model.state_dict(),  
                'epoch_loss_lis': model.epoch_loss_lis,
                'epoch_acc_lis': model.epoch_acc_lis
            },
            file_path
        )
        print(f"[INFO] Model saved to {file_path}")
    
    @staticmethod
    def load(file_path:str, map_location='cpu'):
        if os.path.exists(file_path):
            checkpoint = torch.load(file_path, map_location=map_location)
            model_name = checkpoint['model_name']
            model = model_components.VGG(model_name)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.epoch_loss_lis = checkpoint['epoch_loss_lis']
            model.epoch_acc_lis= checkpoint['epoch_acc_lis']
            print(f"[INFO] Model loaded from {file_path}")
        else:
            print(f"[ERROR] Model file does not exists. Model not loaded.")
        
        return model

    @staticmethod
    def load_models(folder_path: str):
        models = []
        for model_name in os.listdir(folder_path):
            model_folder_path = os.path.join(folder_path, model_name)
            epoch_files = sorted(os.listdir(model_folder_path))
            model_file_path =os.path.join(model_folder_path, epoch_files[-1])
            model = ModelManager.load(model_file_path)
            models.append(model)
        return models
        
    @staticmethod
    def plot_training_loss(epoch_loss_lis: List[float]):
        pass

    @staticmethod
    def plot_training_acc(epoch_acc_lis: List[float]):
        pass

    @staticmethod
    def count_layers(model: model_components.VGG):
        type_to_count = dict()
        for module in model.modules():
            if module.__class__.__name__ in type_to_count:
                type_to_count[module.__class__.__name__] += 1
            else:
                type_to_count[module.__class__.__name__] = 1
        
        return type_to_count

    @staticmethod
    def print_model_summary(model):
        pass
        # layers_count = ModelManager.count_layers(model)
        # print("Layer Information")
        # for layer, count in layers_count.items():
        #     print(f"{layer}: {count}")
        
        # print(f"\n{'='*10}\n")
        # print("Model Summary")


        # print(f"\n{'='*10}\n")
