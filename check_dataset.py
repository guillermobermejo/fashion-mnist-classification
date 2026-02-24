from torchvision import datasets, transforms

print("Checking dataset...")

ds_train = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

ds_test = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print("Train size:", len(ds_train))
print("Test size:", len(ds_test))
