from dataset import UnifiedPoseDataset, HO3D_v2_Dataset, HO3D_v3_Dataset

# dataset = UnifiedPoseDataset(mode='train', loadit=False, name='train')
# print(len(dataset))
# dataset = UnifiedPoseDataset(mode='test', loadit=False, name='test')
# print(len(dataset))

dataset = HO3D_v3_Dataset(mode='train', loadit=False)
print(len(dataset))

