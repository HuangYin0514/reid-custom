# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torchreid
import torch

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datamanager = torchreid.data.ImageDataManager(
    root='../reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=16,
    transforms=['random_flip', 'random_crop']
)



# %%

model = torchreid.models.build_model(
    name='mylinearnnet_baisc',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

model = model.to(device)

# %%
model1 = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=False
)

model1 = model1.to(device)

# %%
input = torch.randn(4,3,256,128)
model1(input).shape

# %%
