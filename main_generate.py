import argparse
import time
from pocket_flow import PocketFlow, Generate
from pocket_flow.utils import *
from pocket_flow.utils import mask_node, Protein, ComplexData, ComplexData
import collections



def str2bool(v):
    if v.lower() in {'yes', 'true', 't', 'y', '1'}:
        return True
    elif v.lower() in {'no', 'False', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pkt', '--pocket', type=str, default='None', help='the pdb file of pocket in receptor')
    parser.add_argument('--ckpt', type=str, default='./ckpt/ZINC-pretrained-255000.pt', help='the path of saved model')
    parser.add_argument('-n', '--num_gen', type=int, default=100, help='the number of generateive molecule')
    parser.add_argument('--name', type=str, default='receptor', help='receptor name')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='cuda:x or cpu')
    parser.add_argument('-at', '--atom_temperature', type=float, default=1.0, help='temperature for atom sampling')
    parser.add_argument('-bt', '--bond_temperature', type=float, default=1.0, help='temperature for bond sampling')
    parser.add_argument('--max_atom_num', type=int, default=40, help='the max atom number for generation')
    parser.add_argument('-ft', '--focus_threshold', type=float, default=0.5, help='the threshold of probility for focus atom')
    parser.add_argument('-cm', '--choose_max', type=str, default='1', help='whether choose the atom that has the highest prob as focus atom')
    parser.add_argument('--min_dist_inter_mol', type=float, default=3.0, help='inter-molecular dist cutoff between protein and ligand.')
    parser.add_argument('--bond_length_range', type=str, default=(1.0,2.0), help='the range of bond length for mol generation.')
    parser.add_argument('-mdb', '--max_double_in_6ring', type=int, default=0, help='')
    parser.add_argument('--with_print', type=str, default='1', help='whether print SMILES in generative process')
    parser.add_argument('--root_path', type=str, default='gen_results', help='the root path for saving results')
    parser.add_argument('--readme', '-rm', type=str, default='None', help='description of this genrative task')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args = parameter()
    if args.name == 'receptor':
        args.name = args.pocket.split('/')[-1].split('-')[0]
    ## Load Target
    assert args.pocket != 'None', 'Please specify pocket !'
    assert args.ckpt != 'None', 'Please specify model !'
    pdb_file = args.pocket
    args.choose_max = str2bool(args.choose_max)
    args.with_print = str2bool(args.with_print)

    pro_dict = Protein(pdb_file).get_atom_dict(removeHs=True, get_surf=True)
    lig_dict = Ligand.empty_dict()
    data = ComplexData.from_protein_ligand_dicts(
                    protein_dict=torchify_dict(pro_dict),
                    ligand_dict=torchify_dict(lig_dict),
                )

    ## init transform
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom(atomic_numbers=[6,7,8,9,15,16,17,35,53])
    focal_masker = FocalMaker(r=6.0, num_work=16, atomic_numbers=[6,7,8,9,15,16,17,35,53])
    atom_composer = AtomComposer(knn=16, num_workers=16, for_gen=True, use_protein_bond=True)

    ## transform data
    data = RefineData()(data)
    data = LigandCountNeighbors()(data)
    data = protein_featurizer(data)
    data = ligand_featurizer(data)
    node4mask = torch.arange(data.ligand_pos.size(0))
    data = mask_node(data, torch.empty([0], dtype=torch.long), node4mask, num_atom_type=9, y_pos_std=0.)
    #data = focal_masker.run(data)
    data = atom_composer.run(data)

    ## Load model
    print('Loading model ...')
    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)

    config = ckpt['config']
    model = PocketFlow(config).to(device)
    model.load_state_dict(ckpt['model'])
    print('Generating molecules ...')
    temperature = [args.atom_temperature, args.bond_temperature]
    # print(args.bond_length_range, type(args.bond_length_range))
    if isinstance(args.bond_length_range, str):
        args.bond_length_range = eval(args.bond_length_range)
    generate = Generate(model, atom_composer.run, temperature=temperature, atom_type_map=[6,7,8,9,15,16,17,35,53],
                        num_bond_type=4, max_atom_num=args.max_atom_num, focus_threshold=args.focus_threshold, 
                        max_double_in_6ring=args.max_double_in_6ring, min_dist_inter_mol=args.min_dist_inter_mol,
                        bond_length_range=args.bond_length_range, choose_max=args.choose_max, device=device)
    start = time.time()
    generate.generate(data, num_gen=args.num_gen, rec_name=args.name, with_print=args.with_print,
                      root_path=args.root_path)
    os.system('cp {} {}'.format(args.ckpt, generate.out_dir))
    
    gen_config = '\n'.join(['{}: {}'.format(k,v) for k,v in args.__dict__.items()])
    with open(generate.out_dir + '/readme.txt', 'w') as fw:
        fw.write(gen_config)
    end = time.time()
    print('Time: {}'.format(timewait(end-start)))
