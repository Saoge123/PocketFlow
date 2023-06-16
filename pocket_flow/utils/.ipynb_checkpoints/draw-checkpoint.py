import os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import  Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from .train import verify_dir_exists




def draw_docked_mol_list(mol_list, molsPerRow=5, subImgSize=(300,300)):
    opts = DrawingOptions()
    opts.atomLabelFontSize = 30
    opts.bondLineWidth = 1.5
    opts.colorBonds = False
    
    legends = []
    for m in mol_list:
        AllChem.Compute2DCoords(m)
        docking_score = 'Docking Score: {:.3f}'.format(float(m.GetProp('r_i_docking_score')))
        mw = 'MolWt: {:.3f}'.format(Descriptors.MolWt(m))
        name = m.GetProp('_Name')
        legend = name + '\n' + mw + '\n' + docking_score
        legends.append(legend)

    img = Draw.MolsToGridImage(
        mol_list,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        legends=legends,
        returnPNG=False
    )
    return img
'''
def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
    #if os.path.isdir(dirname) == False:
        os.makedirs(os.path.dirname(dirname))'''


def HighlightAtomByWeights(mol,         # rdkit.Molecule的instance
                           #weights,     # 注意力系数，注意weights的顺序应该与rdkit.Molecule原子idx的顺序一致
                           save=None, 
                           size=(300,300), 
                           colors=['#FFFFFF','#FF0000'],  # 颜色变化，可以定义多个颜色
                           bondLineWidth=5,   # 化学键显示的宽度
                           FontSize=3,
                           fixedBondLength=50,    # 化学键显示的长度
                           legend=None,
                           legendFontSize=5,
                           elemColor=False,
                           withIsomeric=True):  # 是否按元素给原子着色
    
    draw = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    option = rdMolDraw2D.MolDrawOptions()
    option.bondLineWidth = bondLineWidth
    option.fixedBondLength = fixedBondLength
    option.setHighlightColour((0.95,0.7,0.95))
    option.baseFontSize = FontSize
    option.legendFontSize = legendFontSize
    if elemColor == False:
        option.updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
    draw.SetDrawOptions(option)
    if withIsomeric:
        AllChem.Compute2DCoords(mol)
    else:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        mol = Chem.MolFromSmiles(smi)
    rdMolDraw2D.PrepareAndDrawMolecule(draw, mol, legend=legend)
    draw.FinishDrawing()
    if save:
        if '.png' in save:
            n = 0
            path = '/'.join(save.split('/')[0:-1])
            while os.path.exists(save):
                n += 1
                name = save.split('/')[-1].split('.')[0]
                name = name + '_' + str(n) + '.png'
                save = path + '/' + name
            else:
                draw.WriteDrawingText(save)
    else:
        return draw

'''
supp = list(iter(Chem.SDMolSupplier('dockpose-idscore2500-first50.sdf', removeHs=True, sanitize=1)))
save_path = './img_docking_results/idscore_filtered/'
verify_dir_exists(save_path)
for ix,mol in enumerate(supp[:50]):
    docking_score = 'Docking Score: {:.3f}'.format(float(mol.GetProp('r_i_docking_score')))
    mw = 'MolWt: {:.3f}'.format(Descriptors.MolWt(mol))
    legend = mw + '\n' + docking_score
    name = save_path + '/' + 'No-' + str(ix) + '-' + mol.GetProp('_Name') + '.png'
    HighlightAtomByWeights(mol, save=name, bondLineWidth=2, FontSize=1, elemColor=True, legend=legend, legendFontSize=100)
'''

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def CombineImages(img_file_list, col_num=7, save_dir='./', title='image', img_size=None):
    
    num_img = len(img_file_list)
    num_row = num_img//col_num+1 if num_img%col_num != 0 else num_img//col_num
    
    if img_size is None:
        img_size = Image.open(img_file_list[0]).size
    #print((img_size[1]*col_num, img_size[0]*num_row))
    toImage = Image.new('RGB', (img_size[1]*col_num, img_size[0]*num_row), color=(255,255,255))
    #print(toImage.size)
    '''for y in range(1, num_row+1):
        for x in range(1, num_per_row):'''
    x_cusum = 0
    y_cusum = 0
    num_has_paste = 0
    for img_file in img_file_list:
        img = Image.open(img_file)
        #print((x_cusum, y_cusum))
        toImage.paste(img, (x_cusum, y_cusum))
        num_has_paste += 1
        #print(num_has_paste)
        if num_has_paste%col_num==0:
            x_cusum = 0
            y_cusum += img_size[1]
        else:
            x_cusum += img_size[0]
            
    #toImage.save("merged.png")
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')  # 取消所有边框，坐标轴
    plt.imshow(toImage)
    #plt.title(title, y=0.02) # title默认设置在图像上方，通过设置y改变位置
    #plt.show()
    verify_dir_exists(save_dir)
    toImage.save(save_dir+'/'+title+'.png')
    #plt.savefig(fname=save_dir+'/'+title)
    plt.clf()    # 清除当前图形。如果不清除，当使用循环大量作图，机器会为plt分配越来越多的内存，速度会逐渐变慢。


'''
import glob

l = sorted([(int(i.split('/')[-1].split('-')[1]), i) for i in glob.glob('img_docking_results/simlarity_filtered/*.png')], key=lambda x:x[0])
CombineImages([i[1] for i in l])
'''