
from torchvision.datasets import CIFAR10
test_dataset = CIFAR10(
    "./datasets/cifar10_test",
    train=False,
    download=True
)

img_id = 0
for x, _ in test_dataset:
    x.save(f"./image_cifar10/{img_id}.png")
    img_id += 1