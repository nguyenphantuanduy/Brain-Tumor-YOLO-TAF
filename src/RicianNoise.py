import numpy as np
import albumentations as A

class RicianNoise(A.ImageOnlyTransform):
    def __init__(self, std=0.05, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.std = std

    def apply(self, img, **params):
        # tạo nhiễu Rician
        noise_real = np.random.normal(0, self.std, img.shape)
        noise_imag = np.random.normal(0, self.std, img.shape)
        noisy = np.sqrt((img + noise_real)**2 + noise_imag**2)
        # scale về [0, 255]
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
