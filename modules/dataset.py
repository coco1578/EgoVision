import cv2
import torchvision.transforms as transforms
import albumentations as A

from torch.utils.data import Dataset



class CustomDataset(Dataset):

    def __init__(self, X, y=None, image_size=256, max_size=None, min_size=None, train=False):

        self._X = X
        self._y = y
        self._image_size = image_size
        self._max_size = max_size
        self._min_size = min_size
        self._train = train
        self._transform = ImageTransformer()

    def __len__(self):

        return len(self._X)

    def __getitem__(self, item):

        image_path = self._X[item]
        if self._y:
            label = self._y[item]

        max_x, max_y = self._max_size[item]
        # check size
        max_x = max_x if max_x < 1920 else 1920
        max_y = max_y if max_y < 1920 else 1920
        min_x, min_y = self._min_size[item]
        min_x = min_x if min_x > 0 else 0
        min_y = min_y if min_y > 0 else 0

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[min_y:max_y, min_x:max_x]
        image = cv2.resize(image, (self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC)
        # image = transforms.ToTensor()(image)
        image = self._transform.transform(image, self._train)

        if self._y:
            return image, label
        else:
            return image


class ImageTransformer:

    def __init__(self):

        self._compose = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

        self._augmenter = A.Compose(
            [
                A.OneOf([
                    A.HorizontalFlip(),
                    A.VerticalFlip(),
                    A.RandomRotate90()
                ]),

                A.Cutout(num_holes=30, max_h_size=30, max_w_size=30, fill_value=0)
            ]
        )

    def _vectorize(self, image):

        return self._compose(image)

    def augmentation(self, image):

        return self._augmenter(image=image)['image']

    def transform(self, image, train=True):

        if train:
            image = self._augmentation(image)
        image = self._vectorize(image)

        return image