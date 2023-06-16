from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm
from multiprocessing import Pool
#from pocket_flow.utils import CrossDocked2020

from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerLine2D


BOND_SYMBOL = {'SINGLE':'-', 'DOUBLE':'=', 'TRIPLE':'#', 'AROMATIC':'@'}
def compute_bond_length(mol, atomatic=True):
    #Chem.Kekulize(mol)
    conformer = mol.GetConformer()
    coords = np.array([conformer.GetAtomPosition(a.GetIdx()) for a in mol.GetAtoms()])
    bond_dict = {}
    for b in mol.GetBonds():
        bond_type = b.GetBondType().__str__()
        if b.GetIsAromatic() and atomatic:
            bond_type = 'AROMATIC'
        bond_symbol = BOND_SYMBOL[bond_type]
        a_ix_start = b.GetBeginAtomIdx()
        a_ix_end = b.GetEndAtomIdx()
        bond_lenth = np.linalg.norm(coords[a_ix_start]-coords[a_ix_end])
        #string_bond_lenth = '{:.3f}'.format(bond_lenth)
        
        symbol_start = b.GetBeginAtom().GetSymbol()
        symbol_end = b.GetEndAtom().GetSymbol()
        bond_key = bond_symbol.join(sorted([symbol_start, symbol_end]))
        if bond_key not in bond_dict:
            bond_dict[bond_key] = [bond_lenth]
        else:
            bond_dict[bond_key].append(bond_lenth)
    return bond_dict


def bond_statistic(inputs, from_file=True, interval=5000, n_p=16, sanitize=True):
    BOND_DICT = {}

    for idx in tqdm(range(0, len(inputs), interval)):
        if idx+interval >= len(inputs):
            raw_files = inputs[idx:]
        else:
            raw_files = inputs[idx:idx+interval]

        if from_file:
            mol_list = []
            for sdf in raw_files:
                mol = Chem.MolFromMolFile(sdf[1], sanitize=True)
                if mol:
                    mol_list.append(mol)
        else:
            mol_list = raw_files
        pool = Pool(processes=n_p)
        dict_list = pool.map(compute_bond_length, mol_list)
        for d in dict_list:
            for k in d:
                if k in BOND_DICT:
                    BOND_DICT[k] += d[k]
                else:
                    BOND_DICT[k] = d[k]
    return BOND_DICT

'''
import pickle

BOND_DICT = bond_statistic(cs2020.index, from_file=True, interval=5000, n_p=16)
with open('BondLenthStatistic.pkl','wb') as fw:
    pickle.dump(BOND_DICT, fw)
del BOND_DICT

with open('BondLenthStatistic.pkl', 'rb') as fr:
    BOND_DICT = pickle.load(fr)
'''

class VizBondDensity(object):
    
    def __init__(self, data_bond_dict, gen_bond_dict, linewidth=0.1, color_data="#BC5F6A", color_gen="#19B3B1", 
                 transparency=0.2, legend_size=25, axis_label_size=30, tick_size=25, is_fill=True, 
                 figsize=(14, 9)):
        self.data_bond_dict = data_bond_dict
        self.gen_bond_dict = gen_bond_dict
        self.linewidth = linewidth
        self.color_data = color_data
        self.color_gen = color_gen
        self.transparency = transparency
        self.legend_size = legend_size
        self.axis_label_size = axis_label_size
        self.tick_size = tick_size
        self.is_fill = is_fill
        self.figsize = figsize
    
    def draw(self, bond_type, save_path=None, dpi=300):
        #bond_type = 'O=S'
        density = gaussian_kde(np.array(self.data_bond_dict[bond_type]))
        density.covariance_factor = lambda : .25
        density._compute_covariance()

        density_gen = gaussian_kde(np.array(self.gen_bond_dict[bond_type]))
        density_gen.covariance_factor = lambda : .25
        density_gen._compute_covariance()

        # Create a vector of 200 values going from 0 to 8:
        xs = np.linspace(1, 2, 200)
       
        # Set the figure size
        plt.figure(figsize=self.figsize)
        
        #设置图片的右边框和上边框为不显示
        ax=plt.gca()  #gca:get current axis得到当前轴
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        # 开始画图
        plt.plot(xs, density(xs), color=self.color_data, alpha=self.transparency, linewidth=self.linewidth)
        if self.is_fill:
            plt.fill_between(xs, density(xs), color=self.color_data, alpha=self.transparency, label='Dataset')

        plt.plot(xs, density_gen(xs), color=self.color_gen, alpha=self.transparency, linewidth=self.linewidth)
        if self.is_fill:
            plt.fill_between(xs, density_gen(xs), color=self.color_gen, alpha=self.transparency, label='Generate')
        plt.legend(prop={'size': self.legend_size})#framealpha=1, handler_map={plt.Line2D: HandlerLine2D(update_func=updateline)},
                      #frameon=True, fontsize=20)
        plt.yticks(size=self.tick_size) # fontproperties = 'Times New Roman', 
        plt.xticks(size=self.tick_size) # fontproperties = 'Times New Roman', 
        plt.xlabel('Bond Length (Å)', fontdict={'size':self.axis_label_size}) # 'family': Arial
        plt.ylabel('Density', fontdict={'size':self.axis_label_size})
        if save_path:
            plt.savefig(save_path + '/' + '{}.png'.format(bond_type), dpi=dpi)
        #plt.show()
        return plt