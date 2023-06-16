from weakref import ref
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import QED, Descriptors, Draw
from rdkit import DataStructs
from . import sascorer
import xlsxwriter
from easydict import EasyDict
from io import BytesIO
import glob


SIMILARITY_METRIC = ['DataStructs.TanimotoSimilarity',
              'DataStructs.DiceSimilarity',
              'DataStructs.CosineSimilarity',
              'DataStructs.SokalSimilarity',
              'DataStructs.RusselSimilarity',
              'DataStructs.KulczynskiSimilarity',
              'DataStructs.McConnaugheySimilarity']


class ProcessSmilesFile(object):
    def __init__(self, sdf_file):# refer_mol=None):
        suppl = Chem.SDMolSupplier(sdf_file)
        self.smiles = {}
        #self.refer_mol = refer_mol
        for idx, mol in enumerate(suppl):
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol)
            self.smiles[smi] = {}
            self.smiles[smi]['mol'] = mol
            self.smiles[smi]['index'] = idx
            self.smiles[smi]['name'] = mol.GetProp('_Name')
            self.smiles[smi]['sdf'] = sdf_file
    
    def _QedScore(self,):
        for smi in self.smiles:
            #mol = Chem.MolFromSmiles(smi)
            #print(self.smiles[smi])
            self.smiles[smi]['QED'] = QED.qed(self.smiles[smi]['mol'])
    
    def _SAScore(self,):
        for smi in self.smiles:
            #mol = Chem.MolFromSmiles(smi)
            try:
                self.smiles[smi]['SA'] = sascorer.calculateScore(self.smiles[smi]['mol'])
            except:
                self.smiles[smi]['SA'] = 10.0
    
    def _Lipinski(self,):
        for smi in self.smiles:
            m = self.smiles[smi]['mol']
            #wt = sum([a.GetMass() for a in m.GetAtoms()])
            #self.smiles[smi]['MolWt'] = wt
            self.smiles[smi]['MolWt'] = Descriptors.MolWt(m)
            self.smiles[smi]['LogP'] = Descriptors.MolLogP(m)
            self.smiles[smi]['TPSA'] = Descriptors.TPSA(m)
            self.smiles[smi]['HBD'] = Descriptors.NumHDonors(m)
            self.smiles[smi]['HBA'] = Descriptors.NumHAcceptors(m)
            self.smiles[smi]['RB'] = Descriptors.NumRotatableBonds(m)
            self.smiles[smi]['Lipinski'] = [self.smiles[smi]['MolWt'],
                                            self.smiles[smi]['LogP'],
                                            self.smiles[smi]['TPSA'],
                                            self.smiles[smi]['HBD'],
                                            self.smiles[smi]['HBA'],
                                            self.smiles[smi]['RB']]
    
    def _Similarity(self, refer_mol, metric='DataStructs.TanimotoSimilarity'):
        metric = eval(metric)
        refer_fp = AllChem.GetMorganFingerprint(refer_mol, 2)
        for smi in self.smiles:
            fp = AllChem.GetMorganFingerprint(self.smiles[smi]['mol'],2)
            '''similarity = DataStructs.FingerprintSimilarity(
                fp, refer_fp, metric=metric
                )'''
            similarity = metric(fp, refer_fp)
            self.smiles[smi]['similarity'] = similarity

    def Filter(self, molwt=200, lp_rule=False, refer_mol=None, 
               metric='DataStructs.TanimotoSimilarity', similarity=0.5):
        self._QedScore()
        self._SAScore()
        self._Lipinski()
        if refer_mol:
            self._Similarity(refer_mol, metric=metric)
        #smiles = EasyDict(self.smiles)
        out = {}
        for s in self.smiles:
            smi = EasyDict(self.smiles[s])
            if smi.MolWt < molwt:
                continue
            if lp_rule:
                l = [smi.MolWt<500, smi.LogP<5, smi.RB<10, 
                        smi.TPSA<140, smi.HBA<10, smi.HBD<5]
                #lenth = len(set(l))
                if sum(l) != 6:
                    continue
            if refer_mol:
                if smi.similarity < similarity:
                    continue
            else:
                smi.similarity = '--'
            out[s] = smi
        return out


def Mol2Excel(mol_items, work_book=None, name='result.xlsx', worksheet_name='worksheet'):
    #add_worksheet_name = path.split('/')[-2].split('_')[0].split('sample-')[1]
    if work_book is None:
        workbook = xlsxwriter.Workbook(name)
    else:
        workbook = work_book
    worksheet = workbook.add_worksheet(worksheet_name)
    header = ['Image', 'SMILES', 'QED Score', 'SA Score', 'Similarity', 
                'MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RB', 'sdf file',
                'index']
    header_style = {'bold':1,'valign':'vcenter','align':'center','top':2,
                    'left':2,'right':2,'bottom':2}
    HeaderStyle = workbook.add_format(header_style)
    
    item_style = {'align':'center','valign': 'vcenter','top':2,'left':2,
                    'right':2,'bottom':2,'text_wrap':1}
    ItemStyle = workbook.add_format(item_style)
    worksheet.set_column('A:A', 39)
    worksheet.set_column('B:B', 39)
    worksheet.set_column('C:C', 10)
    worksheet.set_column('D:D', 10)
    worksheet.set_column('E:E', 10)
    worksheet.set_column('F:F', 10)
    worksheet.set_column('G:G', 10)
    worksheet.set_column('H:H', 10)
    worksheet.set_column('I:I', 10)
    worksheet.set_column('J:J', 10)
    worksheet.set_column('K:K', 10)
    worksheet.set_column('L:L', 30)
    worksheet.set_column('M:M', 10)

    for ix_, i in enumerate(header):
        worksheet.write(0, ix_, i, HeaderStyle)

    for ix, items in enumerate(mol_items):
        smi = items[-2]
        info = EasyDict(items[-1])
        img_data = BytesIO()
        AllChem.Compute2DCoords(info.mol)
        img = Draw.MolToImage(info.mol)
        img.save(img_data, format='PNG')
        worksheet.set_row(ix+1, 185)
        worksheet.insert_image(ix+1, 0, 'f', {'x_scale': 0.9, 'y_scale': 0.8, 'image_data':img_data, 'positioning':1})
        worksheet.write(ix+1, 1, smi, ItemStyle)
        worksheet.write(ix+1, 2, info.QED, ItemStyle)
        worksheet.write(ix+1, 3, info.SA, ItemStyle)
        worksheet.write(ix+1, 4, info.similarity, ItemStyle)
        worksheet.write(ix+1, 5, info.MolWt, ItemStyle)
        worksheet.write(ix+1, 6, info.LogP, ItemStyle)
        worksheet.write(ix+1, 7, info.TPSA, ItemStyle)
        worksheet.write(ix+1, 8, info.HBD, ItemStyle)
        worksheet.write(ix+1, 9, info.HBA, ItemStyle)
        worksheet.write(ix+1, 10, info.RB, ItemStyle)
        worksheet.write(ix+1, 11, info.sdf, ItemStyle)
        worksheet.write(ix+1, 12, info.index, ItemStyle)
    if work_book is None:
        workbook.close()

def WriteExcel(result_dir_list, name='result.xlsx', molwt=200, lp_rule=False, sorted_idx=0, refer_mol=None):
    '''
    sorted_ix: 用于排序的属性的index, 0:SA; 1:QED; 2:SA*QED
    '''
    #创建一个工作簿并添加一张工作表，工作表是可以命名的
    #workbook = xlsxwriter.Workbook(name)
    for path in result_dir_list:
        mol_dict = ProcessSmilesFile(path).Filter(
            molwt=molwt, lp_rule=lp_rule, refer_mol=refer_mol, 
            metric='DataStructs.TanimotoSimilarity', similarity=0.2
            )
        #print(mol_dict)
        #L = [(mol_dict[i]['SA'], mol_dict[i]['QED'], mol_dict[i]['SA']*mol_dict[i]['QED'], i, mol_dict[i]) for i in mol_dict]
        L = [(mol_dict[i]['SA'], mol_dict[i]['QED'], mol_dict[i]['similarity'], i, mol_dict[i]) for i in mol_dict]
        if sorted_idx==0:
            L = sorted(L, reverse=False, key=lambda x: x[sorted_idx])
        else:
            L = sorted(L, reverse=True, key=lambda x: x[sorted_idx])
        #print(L)
        #add_worksheet_name = path.split('/')[-2]
        #worksheet = workbook.add_worksheet(add_worksheet_name)
        Mol2Excel(L, name=name)#, worksheet_name=add_worksheet_name)
    #workbook.close()


def WriteExcelAll(result_dir_list, name='result.xlsx', molwt=200, lp_rule=False, 
                  sorted_ix=0, refer_mol=None, similarity=0.2):
    all_mol = {}
    for path in result_dir_list:
        mol_dict = ProcessSmilesFile(path).Filter(
            molwt=molwt, lp_rule=lp_rule, refer_mol=refer_mol, 
            metric='DataStructs.TanimotoSimilarity', similarity=similarity
            )
        for smi in mol_dict:
            if smi not in all_mol:
                all_mol[smi] = mol_dict[smi]
    
    L = [(all_mol[i]['SA'], all_mol[i]['QED'], all_mol[i]['similarity'], i, all_mol[i]) for i in all_mol]
    if sorted_ix==0:
        L = sorted(L, reverse=False, key=lambda x: x[sorted_ix])
    else:
        L = sorted(L, reverse=True, key=lambda x: x[sorted_ix])
    Mol2Excel(L, name=name)



'''refmol_1 = Chem.MolFromSmiles('O=C(/C=C/C1=CC=CC=C1)/C=C/C2=CC=CC=C2')
refmol_2 = Chem.MolFromSmiles('O=C(/C(CCC/1)=C/C2=CC=CC=C2)C1=C\C3=CC=CC=C3')
refmol_3 = Chem.MolFromSmiles('O=S(C1=CC2=NC(C3=CC=CC=C3)=CC=C2C=C1)(NC4=CC=CC=C4)=O')
path_list = [i for i in glob.glob('./outputs/HAT1_2022_08_25*/SMILES.txt') if 'sample-1' not in i]
path_list += ['HAT1_2022_08_26__18_59_39/SMILES,txt']
print(path_list)
WriteExcelAll(path_list, molwt=100, name='HAT1_Wt_more_than_100.xlsx', sorted_ix=0, refer_mol=None)'''