from torchvision import transforms as T

preprocessing_func = T.Compose(
    [
    T.Resize((256, 256)),
    T.ToTensor(),
    ]
)

def preprocess(img):
    img = preprocessing_func(img)
    img = img.unsqueeze(0)
    return img