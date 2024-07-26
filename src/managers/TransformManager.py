from src.common_imports import *
class TransformManager:
    @staticmethod
    def apply_transforms(batch, transform):
        batch["image"] = [transform(image) for image in batch["image"]]
        return batch

    @staticmethod
    def get_train_transform(s_min, s_max, img_width=32, img_height=32):
        if s_min == s_max:
            training_scale = s_min
        else:
            training_scale = random.randint(s_min, s_max)
        
        train_transform = transforms.Compose([
                # Resize the image so that the smallest side is training scale, S
                transforms.Resize(training_scale),

                transforms.Lambda(lambda image: image.convert('RGB') if image.mode != 'RGB' else image),
                
                # Randomly crop a 32x32 patch
                transforms.RandomCrop((img_width,img_height)),
                
                # Randomly change the brightness, contrast, and saturation
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),

                # Convert the image to tensor 
                transforms.ToTensor(),

                # Normalize the image with mean and std (pre-computed on the ImageNet Dataset)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return train_transform


    @staticmethod
    def get_test_transform():
        pass