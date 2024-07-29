from src.common_imports import *
class TransformManager:
    @staticmethod
    def apply_transforms(batch, transform):
        batch["image"] = [transform(image) for image in batch["image"]]
        return batch
    
    @staticmethod
    def get_train_transform_single_scale(scale, img_width=32, img_height=32):
        return transforms.Compose([
                # Resize the image so that the smallest side is training scale, S
                transforms.Resize(scale),
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

    @staticmethod 
    def get_train_transform_multi_scale(s_min, s_max, img_width, img_height):
        return transforms.Compose([
                transforms.RandomResizedCrop((img_width, img_height), scale=(s_min, s_max)),
                transforms.Lambda(lambda image: image.convert('RGB') if image.mode != 'RGB' else image),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    @staticmethod
    def get_test_transform_single_scale(scale, img_width=32, img_height=32):
        return TransformManager.get_train_transform_single_scale(scale, img_width, img_height)