import sys
sys.path.insert(0, 'GPM')
import torch_geometric.transforms as T
from data.dataset.zinc_dataset import ZINC

transform = T.Compose([T.AddRandomWalkPE(8, 'pe')])

print('Processing train...')
ZINC('data/zinc', subset=True, split='train', pre_transform=transform)
print('Processing val...')
ZINC('data/zinc', subset=True, split='val', pre_transform=transform)
print('Processing test...')
ZINC('data/zinc', subset=True, split='test', pre_transform=transform)
print('Done!')