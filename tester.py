import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms

from utils.model import OSnet


def get_image_ref(num):
    if num == 0:
        num = ''
    else:
        num = f"_{num}"

    image = cv2.imread(f'test_img/query{num}.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))

    # t = transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667))
    # t = transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667))
    # t = transforms.Compose([
    #     # transforms.ToTensor(),
    #     # transforms.Lambda(lambda x: (x * 255).type(torch.uint8)),
    #     # transforms.RandomEqualize(p=1),
    #     transforms.Lambda(lambda x: x.type(torch.FloatTensor) / 255.),
    #     transforms.Normalize((0.5355, 0.4852, 0.4441), std=(0.2667, 0.2588, 0.2667)),
    #     transforms.ColorJitter(.1, .1, .1),
    #     transforms.RandomHorizontalFlip(p=.5),
    #     transforms.RandomVerticalFlip(p=.5),
    #     # transforms.TrivialAugmentWide(),
    #     transforms.RandomAffine(30, (.1, .1), (0.75, 1), 30),
    # ])
    # t = Transformation()
    # t = lambda x:x

    ref = cv2.imread(f'test_img/ref{num}.jpg')
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    ref = cv2.resize(ref, (256, 256))

    image = transforms.ToTensor()(image)
    ref = transforms.ToTensor()(ref)
    # image, ref = t(image, ref)

    image = image.unsqueeze(0)
    ref = ref.unsqueeze(0)

    return image, ref


if __name__ == '__main__':
    model_checkpoint = 'log/dtd_dtd_weighted_bce_banded_0.001/snapshot-epoch_2021-11-26-13:15:14_texture.pth'

    model = OSnet()
    model.eval()
    model.load_state_dict(torch.load(model_checkpoint))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if torch.cuda.is_available():
        model = model.cuda()

    for i in range(5):
        im, ref = get_image_ref(i)

        if torch.cuda.is_available():
            im = im.cuda()
            ref = ref.cuda()

        scores = model(im, ref)
        seg = scores[:, 0, :, :]
        thresh = 0.7
        seg[seg > thresh] = 1
        seg[seg <= thresh] = 0

        fix, axes = plt.subplots(1, 3)

        im: np.ndarray = im[0].detach().cpu().numpy()
        ref: np.ndarray = ref[0].detach().cpu().numpy()
        seg = seg.detach().cpu().numpy()

        im = im.transpose([1, 2, 0])
        ref = ref.transpose([1, 2, 0])
        seg = seg.transpose([1, 2, 0])

        # im = to_pil_image(im[0], mode=)
        # ref = to_pil_image(seg[0])

        seg = np.repeat(seg, 3, 2)

        axes[0].imshow(np.uint8(im * 255))
        axes[1].imshow(np.uint8(ref * 255))
        axes[2].imshow(np.uint8(seg[:, :, 0] * 255), 'jet')
        plt.savefig(f'{i}.png')
        # plt.show()
