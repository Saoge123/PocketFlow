import torch
from pocket_flow.gdbp_model import PocketFlow, reset_parameters, freeze_parameters
from pocket_flow.utils import Experiment, LoadDataset
from pocket_flow.utils.transform import *
#from utils.ParseFile import Protein, parse_sdf_to_dict
from pocket_flow.utils.data import ComplexData, torchify_dict
import os
from easydict import EasyDict



protein_featurizer = FeaturizeProteinAtom()
ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
traj_fn = LigandTrajectory(perm_type='mix', num_atom_type=9)
focal_masker = FocalMaker(r=4, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
atom_composer = AtomComposer(
    knn=16, num_workers=16, graph_type='knn', radius=10, use_protein_bond=True
    )
combine = Combine(traj_fn, focal_masker, atom_composer)
transform = TrajCompose([
    RefineData(),
    LigandCountNeighbors(),
    protein_featurizer,
    ligand_featurizer,
    combine,
    collate_fn
])

dataset = LoadDataset('./data/crossdocked_pocket10.lmdb', transform=transform)
print('Num data:', len(dataset))
train_set, valid_set = LoadDataset.split(dataset, val_num=100, shuffle=True, random_seed=0)
dataset[0]
## reset parameters
device = 'cuda:0'
ckpt = torch.load('../path/to/pretrained/ckpt.pt', map_location=device)
config = ckpt['config']
model = PocketFlow(config).to(device)
model.load_state_dict(ckpt['model'])
print(model.get_parameter_number())
keys = ['edge_flow.flow_layers.5', 'atom_flow.flow_layers.5', 
        'pos_predictor.mu_net', 'pos_predictor.logsigma_net', 'pos_predictor.pi_net',
        'focal_net.net.1']
model = reset_parameters(model, keys)
# model = freeze_parameters(model,key)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2.e-4, weight_decay=0, betas=(0.99, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.6, patience=10, min_lr=1.e-5)

exp = Experiment(
    model, train_set, optimizer, valid_set=valid_set, scheduler=scheduler,
    device=device, data_parallel=False, use_amp=False
    )
exp.fit_step(
    1000000, valid_per_step=5000, train_batch_size=2, valid_batch_size=16, print_log=True,
    with_tb=True, logdir='./finetuning_log', schedule_key='loss', num_workers=8, 
    pin_memory=False, follow_batch=[], exclude_keys=[], collate_fn=None, 
    max_edge_num_in_batch=400000
    )
