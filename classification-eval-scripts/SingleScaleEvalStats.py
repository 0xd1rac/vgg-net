class SingleScaleEvalStats:
    def __init__(self, 
                 model_name: str,
                 training_min_scale: int,
                 training_max_scale: int,
                 test_scale: int,
                 top_1_err: float,
                 top_5_err: float,
                 acc: float
                 ):
        self.model_name = model_name
        self.train_min_scale = training_min_scale
        self.train_max_scale = training_max_scale
        self.test_scale = test_scale
        self.is_single_scale_training = training_max_scale == training_min_scale
        self.top_1_err = top_1_err
        self.top_5_err = top_5_err
        self.acc = acc

    def display_stats(self):
        print(f"Model Name: {self.model_name}")
        print(f"Training Min Scale: {self.train_min_scale}")
        print(f"Training Max Scale: {self.train_max_scale}")
        print(f"Test Scale: {self.test_scale}")
        print(f"Is Single Scale Training: {self.is_single_scale_training}")
        print(f"Top-1 Error Rate: {self.top_1_err:.4f}")
        print(f"Top-5 Error Rate: {self.top_5_err:.4f}")
        print(f"Accuracy: {self.acc:.4f}")

