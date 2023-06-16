import sys
import yaml
import time
from pocket_flow import PocketFlow, Generate
from pocket_flow.utils import *
from pocket_flow.utils import mask_node, Protein, ComplexData, ComplexData


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

    args_default = {
        'pocket': None,
        'ckpt': None,
        'num_gen': 10000,
        'name': None,
        'device': 'cuda:0',
        'atom_temperature': 1.0,
        'bond_temperature': 1.0,
        'max_atom_num': 35,
        'focus_threshold': 0.5,
        'choose_max': True,
        'min_dist_inter_mol': 3.0,
        'bond_length_range': (1.0, 2.0),
        'max_double_in_6ring': 0,
        'with_print': True,
        'root_path': 'gen_results',
        }
    '''
    args = {}
    for items in open(sys.argv[0]).read().split('\n'):
        print(items)
        k,v = [i.strip() for i in items.strip().split(':')]
        if k in {i'''
    #print(sys.argv[1])
    #print(open(sys.argv[1]).read())
    #args = Dict(yaml.safe_load(open(sys.argv[1], 'r')))
    args = Dict(yaml.safe_load(open('gen_config.yml', 'r')))
    #print(args)
    for arg in args_default:
        if arg not in args:
            args[arg] = args_default[arg]
    #print('pocket: ', args.pocket is None)
    #args = Dict(args)
    #args = yaml.load(open(sys.argv[0]).read())
    assert args.pocket is not None, 'Please specify pocket !'
    assert args.ckpt is not None, 'Please specify model !'
    if args.name is None:
        args.name = args.pocket.split('/')[-1].split('-')[0]
    '''args.pocket =  'pockets/HAT1/HAT1-pocket10-norm-surf.pdb'
    args.ckpt =  'finetuning_2nd_log/2023-03-01-10-36/ckpt/50000.pt'#'565000.pt'#'./ckpt/255000.pt' 
    args.choose_max = 1
    args.root_path = 'gen_results_new/'
    args.name = 'HAT1'
    args.num_gen = 100'''
    '''
    args.name = 'HAT1' 
    args.root_path = 'gen_results_new' 
    args.device = 'cuda:0'
    args.focus_threshold = 0.5'''

    ## Load Target
    assert args.pocket != 'None', 'Please specify pocket !'
    assert args.ckpt != 'None', 'Please specify model !'
    pdb_file = args.pocket
    

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
    '''if args.device != 'cpu':
        device = 'cuda:{}'.format(args.device)
    else:
        device = args.device'''
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
    #print('cp {} {}'.format(args.ckpt, generate.out_dir))
    #with open(generate.out_dir+'/readme.txt', 'w') as fw:
    #    fw.write('Model copied from {}\n'.format(args.ckpt))
    #if str(args.readme) != 'None':
    gen_config = '\n'.join(['{}: {}'.format(k,v) for k,v in args.__dict__.items()])
    with open(generate.out_dir + '/readme.txt', 'w') as fw:
        fw.write(gen_config)
    end = time.time()
    print('Time: {}'.format(timewait(end-start)))
'''
'python main_generate.py -pkt {} --ckpt {} -n {} --name {} \
    -root_path {} -d {} -at {} -bt {} --max_atom_num {} -ft {} -cm {} --with_print {}'

python main_generate.py -pkt pockets/METTL16/METTL16-pocket10-norm.pdb --ckpt ../PocketFlow/saved_model/finetuning_new-215000.pt -n 100 --name METTL16 -root_path gen_results -d 0 -at 1 -bt 1 --max_atom_num 35 -ft 0.5 -cm True --with_print True
'''

'''def make_cmd(para_dict):
    cmd_temp = 'python main_generate.py -pkt {} --ckpt {} -n {} --name {} --root_path {} -d {} -at {} -bt {} --max_atom_num {} -ft {} -cm {} --with_print {}'
    cmd = cmd_temp.format(
        para_dict.pkt, para_dict.ckpt, para_dict.num, para_dict.name, para_dict.root_path,
        para_dict.device, para_dict.atom_temp, para_dict.bond_temp, para_dict.max_atom_num, 
        para_dict.focus_threshold, para_dict.choose_max, para_dict.with_print
        )
    return cmd

para_dict = Dict({})
para_dict.pkt = 'pockets/METTL16/METTL16-pocket10-norm.pdb'
para_dict.ckpt = '../PocketFlow/saved_model/finetuning_new-215000.pt'
para_dict.num = 10000
para_dict.name = 'METTL16'
para_dict.root_path = 'gen_results'
para_dict.device = 'cuda:0'
para_dict.atom_temp = 1.0
para_dict.bond_temp = 1.0
para_dict.max_atom_num = 35
para_dict.focus_threshold = 0.5
para_dict.choose_max = True
para_dict.with_print = False
cmd_1 = '\n'.join([make_cmd(para_dict) for _ in range(5)])
open('gbp_pocket_flow_with_edge/gen_{}.sh'.format(para_dict.name),'w').write(cmd_1)'''
