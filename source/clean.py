from dataset import HO3D_v2_Dataset, FHAD_Dataset, FreiHAND_Dataset, Obman_Dataset

# dataset = UnifiedPoseDataset(mode='train', loadit=False)
# print(len(dataset))
# dataset = UnifiedPoseDataset(mode='test', loadit=False, name='test')
# print(len(dataset))

dataset = Obman_Dataset(mode='train', loadit=False)
print(len(dataset))

