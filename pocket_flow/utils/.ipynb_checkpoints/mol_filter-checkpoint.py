from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Crippen, Lipinski
from multiprocessing import Pool
from tqdm.auto import tqdm
from . import sascorer


# from https://doi.org/10.1016/j.bmc.2019.115192
ALERT_STRUCTURES = [
    'C#CI', 'C[C&D2]=!@N[!O]', 'CC([C,H])=!@N[!O]', 'N#CC(=O)', 'C(=[O,S])[F,Cl,Br,I]', '[#6][CH1]=O', 
    'COS(=O)(=O)C(F)(F)F', '[C;H1$(C([#6;!$(C=O)])),H0$(C([#6;!$(C=O)])[#6;!$(C=O)])]=[CH1]!@N([#6;!$(C(=O))])[#6;!$(C(=O))]', 
    '[CX4][Cl,Br,I]', '*=C=*', 'c1(N)ccc(C(=O)NC3(=O))c(c3ccc2)c21', 'N!@C=C-C=[C!r]([OH])', 'NC#N', 
    '[#6]C(=O)-!@OC(=[O,N])', 'O=*N=[N+]=[N-]', 'N=[N+]([O-])C', 'c1ccccc1[N!r]=[N!r]c2ccccc2', 
    '[N;R0]=[N;R0]C#N', 'C(Cl)(Cl)(Cl)C([O,S])[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]', 'N[C!r](=O)S', 
    '[Cl]C([C&R0])=N', 'SC(=[!r])S', 'O1CCOCCOCCOCC1', 'O1CCOCCOCCOCCOCC1', 'O1CCOCCOCCOCCOCCOCC1', 
    'N#CC[OH1]', 'c1c([N;!R](~[O;X1])~[O;X1])cc([N;!R](~[O;X1])~[O;X1])cc1', 
    'c1c([N;!R](~[O;X1])~[O;X1])c([N;!R](~[O;X1])~[O;X1])ccc1', 
    'c1c([N;!R](~[O;X1])~[O;X1])ccc([N;!R](~[O;X1])~[O;X1])c1', 'C=[N+]=[N-]', '[N+]#N', 
    '[C!r]=[C&D2][C!r]=[C&D2][C&D3]=O', 'NC(=S)S', 'S1SC=CC1=S', 'S1C=CSC1=C', 'S[C;!$(C=*)]S', 
    'c1cccc(NC(=NC(=[N,S,O])NC(=O)3)C3=N2)c12', 'O=C1NC(=O)c2nc3ccccc3nc2N1', 
    'c1cc(O)cc(OC(=CC(=O)C=C3)C3=C2)c12', 
    '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]1([$([OX1]),$([NH]);!$([*X3])&!$(*-C=O)])[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]([$([OX1]),$(N);!$([*X3])&!$(*-C=O)])[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]1', 
    '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]1([$([OX1]),$([NH]);!$([*X3])&!$(*-C=O)])[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]([$([OX1]),$(N);!$([*X3])&!$(*-C=O)])[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]1', 
    '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]1[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)](O[$(C=O),$(C=N)])[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)][$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)](O[$(C=O),$(C=N)])[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]1', 'C(=O)Onnn', 'c1[n+]([#6])ccn1([#6])', 
    '[#6,#8,#16]-[CH1]=[NH1]', 'C=N([C,S])[C&v4]', 'N=[C!R]=O', 'N=[C!R]=S', 'C=[C!R]=O', 'P(=S)(S)S', 
    '[#3,#11,#12,#13,#19,#20,#26,#27,#28,#29,#30]', 
    '[#6;$([#6]~[#3,#11,#12,#13,#19,#20,#26,#27,#28,#29,#30])]', '[#16]1cc[#16][#6]1=[#16]', 
    '[C&D1]=C[$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]', 
    '[C&D1]#C[$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]', 
    '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*),$([C&X4])]-[C&D2]=!@C[$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]', 
    '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*),$([C&X4])]-[C&D2]#!@C[$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]', 
    '[C&X4]-[C&D2!r]=[C!r][$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]', 
    '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]-[C&D2!r]=[C!r][$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]-[C&X4]', 
    'C1[C&D2!r]=[C!r][$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]-A=A1', 
    'A1[C&D2!r]=[C!r][$([C&D2](C)(=O)),$(C(C)(=O)[!#8]),$(C(C)(=O)[O&D2]),$(C(C)(#N)),$(N(C)(~O)(~O)),$([S&D4](C)(~O)(~O))]-A1', 
    'c1ccccc1C(=O)C=!@CC(=O)!@*', '[F,I,Cl,Br;!v1;!-]', 'N[F,I,Cl,Br]', '[N+!$(N=O)][O-X1]', 
    'c1ccc(n[o,s]n2)c2c1[N;!R](~[O;X1])~[O;X1]', 'c1c([N;!R](~[O;X1])~[O;X1])cc(n[o,s]n2)c2c1', 
    '[#7v5]', '[N!$([N+][O-])]=O', 'C(O)(O)[OH]', '[#8v3]', 'OO', 'c12cccc3c1c4c(cc3)cccc4cc2', 'C=P', 
    'C=!@CC=!@C', 'c1cccc(cc(cccc2)c2c3)c13', 'c1cccc(c(cccc2)c2cc3)c13', 
    '*[SX2][SX2][SX2]*', '[#16+1!$([#16]~[O-])]', 'c1ccc[o+]c1', 'O=S(=O)O[C,c]', 'COS(=O)O[C,c]', 
    'S(=O)(=O)C#N', '[#16][F,Cl,Br,I]', '[SX2H0]!@[N]', 'C1NNC=NN1', '*[CH1](=!@S)', 'SC#N', 
    'P(=S)(-[S;H1,H0$(S(P)C)])(-[O;H1,H0$(O(P)C)])(-N(C)C)', '*1[O,S]*1', '[#16v3,#16v5]', 
    'C(=O)N(C(=O))OC(=O)', 'B(c1ccccc1)(c2ccccc2)c3ccccc3', 'P(c1ccccc1)(c2ccccc2)(c3ccccc3)', 
    '[Si](c1ccccc1)(c2ccccc2)(c3ccccc3)', 'O=C([C!r]=[C!r]C)[!O&!N]', 'O=C([!O&!N])[C!r]=[C&D1!r]', 
    'N=!@N', 'CC(=O)[C&X2]-!@Oc', 'C=[N!r][N!r]=C', 'S1(CCN4(C(CC41)=O))', 'S1(CC=CN5(C(CC51)=O))', 
    '[S;X2]1[CH2][CH]2[CH]([CH]1)[NH]C([NH]2)=O', 'C(=N)-!@N-!@C(=N)', '*[S&D2!r][S&D2!r]*', 
    '*~[N!H0]!@[N!H0]~*', '[C&D2][C&D2][C&D2][C&D2][C&D2][C&D2][C&D2][C&D2][C&D2][C&v4&D1]', 
    'Nc1aaa(N!@=N)aa1', 'C[n+1]2ccccc2', '[N+1&X4;H0]', 'CC(=[S,O])-!@[S&D2]*', 'CC(=S)-!@O*', 
    '[SX2H1]', '[3#1]', 'N1C(=S)SC(=C)C1=O', 'C1([#6](~[!#6]~[#6][!#6][#6]1=,:[!#6])~[!#1;!#6])=[C;!R;H,H2]', 
    'c1ccc(c(c1)[OH])C=N[#7]', '[#6]C(=[!C;!R])C(=[!C;!R])[#6,$(S(=O)=O)]', 
    'O=C2C(=!@N[#7])c1c(cccc1)N2', 'N([C;H2,H3])([C;H2,H3])c1oc([#6]=,:[#7][#7;H][#6]=,:[!#6])[cH][cH]1', 
    'c1(C(=O)[OH])c([NH]N=C)cccc1', 'c1c(c(ccc1)C(cc)=O)[NH2,$([NH1](c)[!$(C=O)])]', 
    'C(C#N)(C#N)C([$(C#N),$(C=N)])C#N', 'C(C#N)(C#N)C([NH2])=CC#N', 'C(C#N)(C#N)=N[NH]c1ccccc1', 
    '[NH2]C1=C(C#N)[CH](cc)C(C#N)=C([NH2])S1', '[#7+](cc)=,:[#6][CH]=CN([CX4])[#6]', 
    'C(C#N)(C#N)=Cc1ccccc1', 'C1(=C)C(N=CS1)=O', 'C1(C(C=C[!C]1)=C)=[!C]', 'C1(=C)C(=O)NNC1=O', 
    '[#7]C=!@C2C(=O)c1c(cccc1)[!C]2', 'c1(ccccc1)[CH]=!@C3C(=O)c2c(cccc2)S3', 'c12ccccc1C(=O)C(=C)C2=O', 
    'C=!@C([!#1])-@C(=!@[!C])-@C(=!@C)[!#1]', '[#6]C(=O)[CH]=C([NH][#6])C(=O)O[#6]', 
    '[#7]~c1[$(nn),$(n[cH]),$(nc[#7;H,H2]),$(c([#7])n),$(c([#7])[cH]),$(c([#7])c[#7;H,H2])][n,cH,$(c[#7;H,H2])]c([#7H,#7H2,$(O[C;H2,H3])])nn1', 
    'c12c([cH][cH][cH][cH]1)[cH]c(c([cH]2)[OH])C(=O)[NH]N=C', 
    '[C;H2,H3]N([C;H2,H3])c1[cH][cH]c([CH]=NN[$(C(=O)[CH2]Scn),$(C(=O)[CH2]aan),$(C(=O)cc[OH]),$(cn),$([CH2][C;H,H2][OH])])[cH][cH]1', 
    '[#6]N1[CH2][CH2]N([CH2][CH2]1)N=[CH]ca', 'C1C(=[!C;!R])C(=O)[!C]*=1', 'C1(C(=O)NC(=O)NC(=O)1)=N', 
    'c12ccccc1C(=O)[CX4]C2=O', 'c1ccc3c2c(cccc12)[NH][CX4][NH]3', 'n2(c1acccc1)c([CX4])c[cH]c2[CX4]', 
    'N1C(=S)S[CX4]C1=O', 'c1c(ccc(c1)[NH]S(=O)=O)[OH]', 
    'c1([CH2,$(cc)])[cH,$(c[CH2,CH3]),$(cC=O)]sc([nH,$(n[CH,CH2,CH3]),$(n(cc)cc)]1)=[N;!R]', 
    'c1([$([#7][#6](=O)cc),NH2])c(-!@C(=O)N[C;H2,H3])sc(n1[$([CH2][CH]=[CH2]),$(cc)])=S', 
    'o1c(sc2cc(ccc12)[#7,#8])=[O,S]', 'S=c1cc[!#6]cc1', '[#6][#6](=S)[#6]', 
    '[NH2]c1sc([!#1])c([!#1])c1C=O', 'c1cccc2c1cc3c(n2)nc4c(c3[#7])cccc4', 
    '[NH2]c1c([NH]S(=O)=O)[cH]c([NH][C;H2,H3])c([F,Cl,Br,I])[cH]1', 
    'c13[cH][cH][cH][cH]c1oc2[cH]c([NH][C;H2,H3])c(O[C;H2,H3])[cH]c23', 
    'c1ccccc1N3C(=O)C([NH]c2ccc(O[C;H2,H3])cc2)=C([F,Cl,Br,I])C3=O', 
    '[C;H2,H3]Oc1[cH][cH][cH][cH]c1[NH][CH](C=O)S', '[NH2]c1c([OH])cc(S(=O)(=O)[OH])cc1', 
    'c2([NH2])c([OH])c(C=O)[cH]c1[cH][cH][cH][cH]c12', 'c13c(C(Nc2c1cccc2)([#6])[#6])ssc3=*', 
    'c1c(ccc([NH2,$([NH][CX4]),$([N]([CX4])[CX4])])c1)[CX4]c2ccc(cc2)[NH2,$([NH][CX4]),$([N]([CX4])[CX4])]', 
    '[C;H2,H3]N([C;H2,H3])c1[cH][cH]c([CH]=NN=C([#6])cc)[cH][cH]1', 
    '[cH]1c([cH]nc2c1[cH][cH][cH][cH]2)[CH2]N3c4c([CH2][CH2]3)[cH][cH][cH][cH]4', 
    '[cH]2c([NH]C(=S)[NH]c1ccccc1)[cH]c(N([C;H2,H3])[C;H2,H3])[cH][cH]2', 
    'N3C=C(C=O)C(c1ccc(N([C;H2,H3])[C;H2,H3])cc1)[#6]2=,:[#6]3~[#7]~[#6](~[#16])~[#7]~[#6](~[#7])~2', 
    '[C;H2,H3]N([C;H2,H3])c1[cH][cH]c(o1)[CH]=CC#N', 
    '[NH2]c2c(N=C1[CH]=[#6]~[#6]~[#6]=C1)[cH][cH][cH][cH]2', 
    '[NH2]c1[cH][cH]c([cH][cH]1)c2[cH]c(C=O)c([C;H2,H3])o2', 
    'c1(O[C;H2,H3])[cH,$(c[C;H2,H3])][cH]c(O[C;H2,H3])c([CH2][$([NH]C(=O)[CH2][CH2][C;H2,H3]),$([CH]([C;H2,H3])[NH]C(=S)[N;H,H2])])[cH]1', 
    'c2(c([NH2])n(c1c(cccc1)C(=O)[OH])nc2C=O)[$(C#N),$(C=S)]', 'c12ncccc1c([NH2])c(C(=O)~[OX1])s2', 
    'C3(=O)C(=[CH][NH]c1c(C(=O)[OH])cccc1)N=C(c2ccccc2)O3', 'c1([OH])ccc([NH]C(=O)cc)c(c1)C(=O)[OH]', 
    'c1(oc([#6])[cH][cH]1)C(=O)[NH]c2[cH][cH][cH][cH]c2C(=O)[OH]', 
    'c1ccccc1C(=O)[NH]c2ccccc2C(=O)[NH][NH]c3nccs3', 'c1ccc2c(cc1)ccc2', 'c1(c(ccccc1)[N;H,H2])=N[#6]', 
    'c1(ccc(cc1)c2cc(c3c(c(c2)[OH])coc3)=O)S[C;H2,H3]', 'c13ccccc1sc(=NN=c2cccccc2)n3[C;H2,H3]', 
    'c23c([Br])cc1c(cco1)c2[cH]cc(=O)o3', 
    'c12c([F,Cl,Br,I])cc([F,Cl,Br,I])cc1[cH]c(C(=O)[NH2])c(=[NH])o2', 
    'c1(c(nc2c(c1C#N)c(c(cc2)C#N)[NH2])[NH2])C#N', 
    'n3c([SX2]c1c([NH2])cccc1)c(C#N)c(c2ccccc2)c(C#N)c3[NH2]', 'C1(C#N)(C#N)[CH](C(=O)[#6])[CH]1', 
    'C=CC(C#N)(C#N)C(C#N)=C[NH2]', 'ccC(=O)[NH]C(=O)C(C#N)=[CH][NH]cc', 'O=S(=O)C(C#N)=N[N;H,H2]', 
    'c2(c1ccccc1nnc2)C(cc)C#N', 'C1(C([C;H2,H3])=C(C#N)[#6](~[#8])~[#7]~[#6]1~[#8])=[CH]cc', 
    'O=c1c(cc(nn1)C=O)C#N', 'n2(c1ccccc1)c(=O)c(C#N)cc(C#N)n2', 
    '[NH2]C1=C(C#N)[C;H,H2](cc)C([C;H2,H3])=C(C=C)O1', 
    '[NH2,$([OH])]C2=C(C#N)[CH](cc)c1c(n([#6])nc1)O2', '[NH2]C1=C(C#N)[CH](cc)C(C#N)=C(cc)O1', 
    '[NH2]C2=C(C#N)[CH](cc)c1c(ccs1)O2', '[C;H2,H3][SX2]C1=C(C#N)[CH](cc)C(C#N)C(=O)N1', 
    '[NH2]C2=C(C#N)[CH](c1cccs1)C(C(=O)O[#6])=C([C;H2,H3])O2', 
    '[SX2]1C=C(C#N)C([#6])(C=O)C([$(C=O),$(C#N)])=C1[NH2]', '[NH2]C1=C(C#N)[CH](cc)S[CX4]S1', 
    '[NH,$([N][CX4])]1[#6]:,=[#6]([#6](=O)ccc)C([#6])C([$(C=O),$(C#N)])=C1[CH3]', 
    'N2=Nc1n[!#6]nc1N=Ncc2', '[CH]2([OH])c1n[!#6]nc1[CH]([OH])C=C2', 
    'c1ccccc1N([C;H2,H3])[CH]=[CH]C=!@[CH][CH]=CC=@Nc2ccccc2', 'C2(c1c(cc(c(n1)[NH2])C#N)C(C=2)=C)C#N', 
    'C(C#N)(C#N)=C(S)S', '[OH]C(=O)c1c([OH])[cH]c([cH][cH]1)c2[cH][cH]c(o2)[CH]=C(C#N)c3nccn3', 
    'n2([#6])[cH][cH][cH]c2[CH]=C(C#N)c1nccs1', 
    'C3(=O)C(=[CH][$(c1ccccc1),$(c2ccc[!#6]2)])N=C(aaa)[S,$([N]aa)]3', 'N1=CC(C(N1)=S)=C', 
    'c1(ccco1)[CH]=!@C3C(=O)c2c(cccc2)[!C]3', 'S=c1[nH]ccc2c1C(=O)OC2=[C;H,H2]', 
    '[#6]1[O,S]C(C([#6]=,:1)=O)=CC=O', 'C2(=O)C(=[CH]c1cc([F,Cl,Br,I])ccc1O[C;H2,H3])N=C(S[C;H2,H3])S2', 
    '[cH]2[cH][cH][cH]c1[CH2]C(=O)C(=C([C;H2,H3])[C;H2,H3])c12', 'C1C(OC(C=1)(O)[#6])O', 
    'c12[cH]c(O[C;H2,H3])c(O[C;H2,H3])[cH]c1C([#6])=C([#6])S[CH2]2', 
    'c2(O[C;H2,H3])c(O[C;H2,H3])[cH]c1C=C[CH]Sc1[cH]2', 'C1(=O)C(C(C#N)=[CH][#7])C([#7])C=C1', 
    'c23ccc1ccccc1c2[C;H,H2][CX4]NC3=[CH]C(=O)N([C;H2,H3])[C;H2,H3]', 'O=CC1=C(SC(=[CH][#6])S1)C=O', 
    'C1(C(=O)[CH2]C[CH2]C(=O)1)=C([N;H,H2])C=O', 'C#CC(=O)C#C', 
    'aa[CH,$(CC#N)]=C1[#6]:,=[#6]C(=[O,$([N;!R])])[#6]:,=[#6]1', 
    'S1C(=Ncc)[NH,NH2,$(N[CH2][CH2]O),$(Ncc)]C(=O)C1=[CH][$(ccc[Cl]),$(c[!#6])]', 
    'S1C(=O)NC(=S)C1=[CH]cc', 'n2ccc([CH]=C1C(=O)NC(=[!C])N1)c2[C;H2,H3]', 
    '[OH]C(=O)c1cccc(c1)cac[CH]=C2C(=[!C])NC(=[!C])[!C]2', 'n12c(nc(=O)c([C;H2,H3])n1)sc(=[CH]cc)c2=O', 
    'N2([C;H2,H3])C(=S)[NH]C(=[CH]c1cc(Br)ccc1)C2=O', 
    '[C;H2,H3]N2C(=[S,N])[!C]C(=C1[CH]=[CH]ccN1[C;H2,H3])C2=O', 
    'c1ccccc1C(=O)[CH]=c3c(=O)[nH]c(=O)c(=[CH]c2ccccc2)[nH]3', 'O=C1cc[CH2]NC1=[C;H,H2]', 
    'c1(oc([C;H2,H3])[cH][cH]1)[CH]([OH])C#C[CX4]', 'ccn2nc1[CH2][S;X2][CH2]c1c2[NH]C(=O)[CH]=[C;H,H2]', 
    '[cH]1[cH]n([C;H2,H3])c3c1c2[cH][cH]n(c2c(c3O[C;H2,H3])O[C;H2,H3])[C;H2,H3]', 
    'C2(=O)C(=C([C;H2,H3])[NH][CH2][CH2][C;H2,H3])N=C(c1ccccc1)O2', 
    'n4(c1ccccc1)[cH][n+](c2ccccc2)c(=Nc3ccccc3)n4', 'n1nc(c2-c(c1C)ccc2)C', 
    's1c2c(c([C;H2,H3])c1[C;H2,H3])c(ncn2)NN=Cc3ccco3', 'C2([#6])[CH2]C(=O)n1nc([NH2])c([NH2])c1N=2', 
    'c2(c1n(c(ccn1)=O)nc2c3nccc3)C#N', 'n2nnnc1cccc-12', 
    'c2([CH2][C;H2,H3])c([C;H2,H3])c1c(=[X1])n([C;H,H2][$(C(=O)O),$(cc)])[cH,$(cS[C;H2,H3])]nc1[a;X2]2', 
    'c1ccc2c(c1)nc3c(n2)ccc4c3cccc4', 'n1c3c(c(=N)c2ccccc12)cccc3', 
    'c23cc1ccccc1cc2n([C;H2,H3])c(=O)c(cc[NH][C;H2,H3])n3', 
    'c12acccc1nc([CH]=C([OH])[#6])c([CH]=C([OH])[#6])n2', 
    'c1ccc2c(c1)nc(c(n2)[CH2]C(=O)cc)[CH2]C(=O)cc', 'c1ccc2c(c1)nc(c(n2)c3ccccc3)c4c(cccc4)[OH]', 
    '[C;H2,H3]Oc1[cH][cH][cH][cH]c1[NH]c3c2c(O[C;H2,H3])ccc(O[C;H2,H3])c2ncc3', 
    'N=c1[nH]c([N;H,H2,H3])c([N;H,H2])nn1', '[NH](c1[cH][cH][cH][cH]c1[OH])c2nnc(=N)oc2[NH2]', 
    's1[cH][cH][cH]c1C4[NH]N=C(c2ccncc2)c3c(cccc3)N=4', 
    'C([F])([F])C(=O)[NH]c1[cH]n([CH2][CH2]O[CH2]cc)n[cH]1', 'c12cccca1c([!#1])a[n+](~cc)a2', 
    '[#6]13~[#7](cc)~[#6]~[#6]~[#6]~[#6]~1~[#6]2~[#7]~[#6]~[#6]~[#6]~[#7+]~2~[#7]~3', 
    'C1(C=O)(cc)[SX2]C=N[NH]1', 'S=c2[nH]nc(c1[cH][cH]c(O[C;H2,H3])[cH][cH]1)o2', 'N=C1SC(=N)N=C1', 
    'N1([C;H2,H3])C(=S)N(cc)C(=Ncc)C1=Ncc', 'S1C(=N[NH])SC(=Ncc)C1=Ncc', 
    'c1ccccc1C4=Nn3c2ccccc2[n+]c3S[CX4]4', 'n2c1ccccc1n([C;H2,H3])c2S[CH2]C(=O)[NH]N=[CH][CH]=[C;H,H2]', 
    'c12ccccc1[CH2][CH2]N=C2[SX2][CH2]C(=O)c3ccccc3', 
    'c1ccc3c(c1)Sc2[cH][cH][cH,$(cO),$(c[SX2]),$(c[CX4]),$(c[NH2,$([NH][CX4]),$([N]([CX4])[CX4])])]([cH]c2C(C3)[NH2,$([NH][CX4]),$([N]([CX4])[CX4])])', 
    '[C;H2,H3][SX2]c2nc1OC=Nccc1nn2', '[C;H2,H3][SX2]c3nc(c1o[cH][cH][cH]1)c(c2o[cH][cH][cH]2)nn3', 
    '[C;H,H2,H3]C3=NN(c1ccccc1)S2[!C]*=CC=23', 'N=c1ncns1', 'cc[NH]C(=O)c1nnsc1[NH]cc', 'n2sc1CNccc1c2=S',
    'C1(SC(=[CH]cc)N(cc)N=1)C=O', 'c1([OH])ccc([OH])c(c1)C(=!@C[#7])C=O', 
    'c14[cH][cH]c(O[C;H2,H3])[cH]c1C(=N[NH]c2[cH][cH]c(C(=O)[OH])[cH][cH]2)c3[cH]c(O[C;H2,H3])[cH][cH]c34', 
    '[OH]C(=O)c1ccc(cc1)NN=[CH]c2ac([cH][cH]2)c3ccccc3', 
    'c1(o[cH,$(c[C;H2,H3])][cH][cH]1)C(=O)[NH]N=[CH,$(C[C;H2,H3])]c3cc(***c2occc2)ccc3', 
    'c1cccc2c1cc4c(c2C=N[NH]c3ccccc3)cccc4', 
    'c1(o[cH,$(c[C;H2,H3])][cH][cH]1)[CH,$(C[C;H2,H3])]=N[NH]c2nccs2', 
    'c1(o[cH,$(c[C;H2,H3])][cH][cH]1)[CH,$(C[C;H2,H3])]=N[NH]c2ccncc2', 
    'c1ccccc1N(c2ccccc2)N=[CH]c3ac([cH][cH]3)c4cc(C(=O)[OH])ccc4', 
    '[OH]C(=O)c1cc(ccc1)cacC=N[NH]C(=O)[CH2]O', 
    'C(c1[cH]c[cH,$(c[CX4])][cH][cH]1)(c2[cH][cH][cH,$(c[Cl])][cH][cH]2)=[$(NO[CH2][CH2][CH2]N([C;H2,H3])[C;H2,H3]),$(NO[CH2][CH2]N([C;H2,H3])[C;H2,H3]),$(N[NH]C(=[NH])[NH2]),$([CH][#7])]', 
    'c12[cH][cH][cH][cH]c1[!#6][cH,$(c[OH]),$(c[C;H2,H3])]c2[CH]=N[NH][$(c3nc[cH]s3),$(c[cH][cH]),$(cncncn),$(cnnnn)]', 
    'c1(s[cH][cH][cH,$(c[C;H2,H3])]1)[CH]=N[NH]c2ccccc2', 
    '[CH]4(n1[cH]n[cH][cH]1)c2[cH]c([Br])[cH][cH]c2[CH2][CH2]c3[cH][cH][cH][cH]c34', 
    'n3c(c1ccccc1)c(c2ccccc2)n(N=!@C)c3[NH2]', 'cc[CH]=[CH][CH]=NN([CX4])[CX4]', 
    'C1(C=Nc2c(N1)cccc2)=[CH]C=O', 'c1cC(=O)C=C1N=[CH]N([CX4])[CX4]', 'c1ccc2c(c1)C(C=N2)=[N;!R]', 
    'cc[CH]=[CH][CH]=NN=C', 'N([C;H2,H3])([C;H2,H3])[CH]=NC([C;H2,H3])=NN([C;H2,H3])cc', 
    'c12c([cH][cH][cH][cH]1)c(c([cH][cH]2)C(=Ncc)[C;H2,H3])[OH]', 
    '[NH](c1ccccc1)N=C(C(=O)[C;H2,H3])[NH][NH,NH2,NH3,$(cc)]', '[N;!R]=C2C(=O)c1c(cccc1)S2', 
    'cc[N;!R]=C2C(=[!C])c1c(cccc1)N2', 'C1(=[!C])C(N=CS1)=O', 'C=[N;!R]c1c([OH])cccc1', 
    '[CX4]1C(=O)NNC1=O', 'O=CC=[CH][OH]', '[C;H2,H3]C([OH])=C(C(=O)[C;H2,H3])[CH]C#C', 
    'cc[NH]N=C([C;H2,H3])[CH2]C([C;H2,H3])=N[NH]cc', 'c1cccc2c1C(c3c2nacc3)=O', 
    'c1cccc2c1C(C3C2=N*=CC3)=O', 'c1(cc3c(c2ccccc12)c4c(C3=O)cccc4)[OH]', 
    'c1c(cccc1)C(=O)[NH]N=C3c2c(cccc2)c4c3cccc4', 'c12c(O[CH2]N(ccO[C;H2,H3])[CH2]1)[cH]cc[cH]2', 
    'c1([NH]C(=O)[CH2][CH2]cc)[cH][cH]c([C;H2,H3])c([NH]C(=O)[CH2][CH2]cc)[cH]1', 
    '[N;H,H2](c1[cH][cH]c(O[CH3])c(O[C;H,H2,H3])[cH]1)C(=O)[NH][CH2][CH2][CH2]N([CH3])cc', 
    'c1c3c(cc2c1cccc2)nc(n3)COC(=O)c4cc(cc(c4)[NH2])[NH2]', 
    'n2(c1[cH][cH][cH][cH]c1C(=O)[NH][CH]([C;H2,H3])[CH2]Occ)[cH][cH][cH][cH]2', 
    'C1(cc)[CH2][CH](C(=O)[#6])[CH](C(=O)[OH])[CH2]C(cc)=1', 'C(cc)(cc)(cc)SccC(=O)[OH]', 
    'c1(C([CX4])([CX4])[NH]C(=O)N([CH2][C;H2,H3])[CH2][CH2][C;H,H2][CH2]cc)[cH][cH][cH]c(C(=[CH2])[C;H2,H3])[cH]1', 
    'c1cc3c2c(c1)cccc2C(=CC3=O)O[C;H2,H3]', 'c13ccc(c2c1c(ccc2)C(C=C3C([F])([F])[F])=O)[#7]', 
    'c1(ccc3c2c(cccc12)C(C=C3)=N)[#7]', 'C(c1ccc([OH])cc1)(c2ccc([OH])cc2)OS(=O)=O', 
    'n12cccc2C=N([#6])CC1', 'n2([C;H2,H3])c1c(ccC(=O)1)cc2[C;H2,H3]', 
    'c1(c2c(c(n1C(O)=O)[C;H2,H3])S[CH2]S2)[C;H2,H3]', 'n2(c1ccccc1)[cH][cH][cH]c2C=N[OH]', 
    'c1ccc2c(c1)c4c3c(C2=O)cccc3no4', 'O=C2c1ccccc1c3c([OH])c(=O)nc4cccc2c34', 
    'C1([#6]:,=[#6][#6]:,=[#6]C1=[!C])=[!C]', 'N2(c1ccccc1)C(=NC=O)S[CH2]C2=O', 
    'N4(c1ccccc1)C(=O)S[CH]([NH]c3c2ccccc2ccc3)C4=O', 'N2(c1ccccc1)C(=O)S[CH2]C2=S', 
    'N3(C(=O)c1ccccc1)C(=Nc2ccccc2)S[CH2]C3=O', 'O=C4CCC3C2C(=O)CC1CCCC1C2CCC3=C4', 
    'C2Cc1ccccc1C(c3c2cccc3)=C[#6]', 'c1ccc3c(c1)[SX2,CX4]c2cc[cH,$(c[Cl]),$(c[CX4])]cc2C3=C[#6]', 
    'c1ccc2c(c1)C(c3c(SC2)scc3)=C', 'c1ccc2c(c1)C(c3c2ccc(c3)[NH2])=[CH][#6]', 
    '[CH3]c2nc([NH]S(c1[cH][cH]c([cH][cH]1)O[CH2][CH2][C;H2,H3])(=O)=O)[cH][cH][cH]2', 
    'c1([cH][cH]c([cH][cH]1)[NH2])S(=O)(=O)[NH]c2[cH][cH][cH]nn2', 
    '[CH3]C([CH3])([CH3])c1[cH]c(C([CH3])([CH3])[CH3])c(O[C;H,H2]N)c[cH]1', 
    'a1aaa(aa1)[CH]=[CH]C([NH][NH]c2n(nnn2)[#6])=O', 'c1([F,Cl,Br,I,$([n+](c)c)])c(-!@C=N)sc(n1)=O', 
    'c1(S[C;R])c([CH]([#6])[#6])sc([nH,$(n[C;H2,H3])]1)=O', 
    's2c([NH]c1c([CH2,CH3,$(cc)])cccc1)[n+]([C;H2,H3])c([#6])[cH]2', 'c1nc(sc1)NNS(=O)=O', 
    'c1c(cc(C(=O)[OH])cc1)[NH]c2nc([cH]s2)c3ccc([CH]([C;H,H2,H3])[C;H,H2,H3])cc3', 
    '[C;H2,H3][NH]C=N[NH]c1nc(cc)[cH]s1', 'O=S(=O)(cc)[NH][NH]c1nc(cs1)cc', 
    'n1c3c(cc2c1nc(s2)[#7])sc(n3)[#7]', 's2ccc(c1csc([NH2])n1)[cH]2', 
    'c2([NH]cc[C;H2,H3])nc(c1ccncc1)[cH]s2', 
    '[C;H2,H3]Oc1[cH][cH]c(O[C;H2,H3])[cH]c1[NH]c3scc(c2ccc(O[C;H2,H3])cc2)n3', '[#6][CH](=S)', 
    '[cH]1coc([cH]1)C(=S)N2[CH2][CH2]*[CH2][CH2]2', '[CX4][NH]C(cc)=[CH]C(=S)[NH]c1ccccc1', 
    '[CH](c1ccccc1)(c2ccccc2)C(=S)[N;H,H2]', 'c(N([C;H,H2,H3])[C;H,H2,H3])c[NH]C(=S)[C;H,H2,H3]', 
    'c1cccnc1C(=S)[NH]c2ccccc2O[C;H2,H3]', 'acC(=S)[NH][NH]ca', '[C;H2,H3]SC(=S)[NH][CH2]cc', 
    'C1[CX4]SC(NC=1)=S', 'O=c2sc1cccc(O[C;H2,H3])c1o2', '[NH]1C(=S)[CH](C#N)[CH](cc)[CH]=C1cc', 
    'S=CC([C;H2,H3])=C([C;H2,H3])N([C;H2,H3])[C;H2,H3]', 'C1=CN(C(c2ccccc12)(C#N)C(=S)S)C=O', 
    'c1c(sc(cc1)=S)[#7]', 'ccC(=[SX1])[SX2][C;H,H2][CH2,CH3,$(cc)]', 'n1c(=O)[cH]c([#6])sc1=S', 
    'c1ccccc1N3C(=O)C(Sc2ccccc2)=[CH]C(=O)3', '[CX4][N+]([CX4][OH])=CS[C;H,H2,H3]', 'S=Cc2n1ccccc1cc2', 
    'c2(nanc(c1o[cH][cH][cH]1)n2)S[CX4]', 'C(=O)(N1CCSCC1)c2c([cH][cH][cH][cH]2)S[C;H2,H3]', 
    'c2c([NH]C(=S)[NH][CH2][CH2][CH2]N([C;H2,H3])c1ccccc1)cccc2', 
    'c2c([NH]C(=S)[NH][CH2][CH2]N([C;H2,H3])c1ccccc1)cccc2', 
    'c1(ccccc1)[NH]C(=S)N[NH]C(=O)c2a[!#6]cc2', 'c1(ccccc1)[NH]C(=S)N[NH]c2ccccc2', 
    'C1[NH][NH]C(=S)N[NH]1', 'c1(ccccc1)[NH]C(=S)N[NH][#6](=,:[#7;R])[#7;R]', 
    'c1(ccccc1)[NH]C(=S)[NH]N=Cc2ccnc2', 'c1(ccoc1[C;H,H2,H3])C=N[NH]C(=S)[N;H,H2]', 
    'c2(=S)n1ccnnc1n[nH]2', '[CX4][SX2]C(=N-aaaa)[NH]N=C', 
    'ccN([C;H2,H3])[CH2][CH2][CH2][NH]C(=S)[NH]c1c([F,Cl,Br,I])[cH]c([C;H2,H3])[cH][cH]1', 
    '[cH]3c([NH]C(=S)[NH][C;H,H2]c1oc([C;H,H2,H3])[cH][cH]1)[cH]c2c(O[CH2]O2)[cH]3', 
    'O=C-!@n2ccc1c2[nH]c(=S)[nH]1', 
    'c12c([cH][cH][cH][cH]1)c([cH][cH][cH]2)C([C;H2,H3])=N[NH]C(=S)[NH]ccc', 
    'c1(ccccc1)[#7;H][#6](=S)[#7][#7;H][#6;H]=,:[#6;H][#6]=O', 
    '[C;H2,H3]N([C;H2,H3])[CH]=CC(=O)c1c([SX2])sc([$(C#N),$(C=O)])c1', 
    's1[cH][cH]c(O[C;H2,H3])c1C(=O)[NH][N;H,H2]', 
    'c1(s[cH,$(c[C;H2,H3])][cH][cH]1)[CH,$(C[C;H2,H3])]C(=O)[NH]c2nccs2', 
    'c1csc(c1[NH2])[CH]=[CH]c2cccs2', '[NH2]c3sc([NH]C(=O)c1ccccc1)c(C#N)c3c2aaaaa2'
    ]  # '([#7+1!r].[#7+1!r].[#7+1!r])'

'''l = []
lines = open('/export/home/jyy/gbp_pocket_flow_with_edge/pocket_flow/utils/alert_struct.csv').read().split('\n')
for line in lines[2:]:
    if ',"' in line:
        l.append(line.split(',"')[-1].strip('"'))
    else:
        l.append(line.split(',')[-1])'''

HAT1_REF_MOL = [
    'CN1C2=C(C=CC(S(=O)(NC3=CC=C(C4=NC=CO4)C=C3)=O)=C2)CCC1C5=CC=CC=C5',
    'CN1C2=C(C=CC(S(=O)(NC3=CC=C(Br)C=C3)=O)=C2)CCC1C4=CC=CC=C4',
    'CN1C2=C(C=CC(S(=O)(NC3=C(C=CC=C4)C4=C(Br)C=N3)=O)=C2)CCC1C5=CC=CC=C5',
    'O=S(C1=CC2=C(C=C1)C=CC(C3=CC=CC=C3)=N2)(NC4=CC=C(Br)C=C4)=O',
    'O=S(C1=CC2=C(C(F)=C1)C=CC(C3=CC=CC=C3)=N2)(NC4=CC=C(Br)C=C4)=O',
    'CN1C2=C(C=CC(S(=O)(NC3=CC=C(C4=COC=N4)C=C3)=O)=C2)CCC1C5=CC=C(Cl)C=C5',
    'CN1C2=C(C=CC(S(=O)(NC3=CC=C(C4=CSC=N4)C=C3)=O)=C2)CCC1C5=CC=C(Cl)C=C5',
    'BrC1=C(O)C=CC(/C=C2CCC/C(C\2=O)=C\C3=CC=C(O)C(Br)=C3)=C1',
    'O=S(C1=CC2=C(NC(C3C2C=CC3)C4=C(C=CC=C4)F)C=C1)(NC5=CC=C(C(O)=O)C=C5)=O',
    'BrC1=C(O)C=CC(/C=C2CC(CC)C/C(C\2=O)=C\C3=CC=C(O)C(Br)=C3)=C1'
    'O=C1/C(CC/C1=C\C2=CC(Br)=C(O)C=C2)=C/C3=CC=C(O)C(Br)=C3',
    'O=C(/C=C/C1=CC(Br)=C(O)C=C1)/C=C/C2=CC=C(O)C(Br)=C2'
            ]

def check_alert_struct(mol):
    Chem.GetSSSR(mol)
    for alert_smarts in ALERT_STRUCTURES:
        p = Chem.MolFromSmarts(alert_smarts)
        if mol.HasSubstructMatch(p):
            return False
    else:
        return True


def check_lipinski(mol):
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    #tpsa = Descriptors.TPSA(mol) # tpsa<=140
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    #rb = Descriptors.NumRotatableBonds(mol)
    if sum([mw<=700, logp<=5, hba<=10, hbd<=5]) != 4:
        return False
    else:
        return True


class Filter(object):
    def __init__(self, sa_threshhold=5.0, qed_threshhold=0.3, mw_threshhold=300):
        self.sa_threshhold = sa_threshhold
        self.qed_threshhold = qed_threshhold
        self.mw_threshhold = mw_threshhold
    
    def _filter(self, mol):
        try:
            qed_check =  QED.qed(mol) > self.qed_threshhold
            sa_check = sascorer.calculateScore(mol) < self.sa_threshhold
            mw_check = Descriptors.MolWt(mol) > self.mw_threshhold
            lip_check = check_lipinski(mol)
            if qed_check and sa_check and mw_check and lip_check:
                struct_check = check_alert_struct(mol)
                if struct_check:
                    return (Chem.MolToSmiles(mol), mol)
                else:
                    return False
            else:
                return False
        except:
            return False
    
    def run(self, mol_list, save_name='Filter_Result.sdf', mol_name=None):
        num_mol = 0
        smi_dict = {}
        #filter(lambda x:x!=False, filter_list)
        for ix, m in tqdm(enumerate(mol_list)):
            m_ = Chem.MolFromSmiles(Chem.MolToSmiles(m))
            out = self._filter(m_)
            if out != False and out[0] not in smi_dict:
                smi_dict.setdefault(out[0])
                with open(save_name, 'a') as sdf_writer:
                    #mol = out[1]
                    if mol_name:
                        m.SetProp('_Name', mol_name)
                    mol_block = Chem.MolToMolBlock(m)
                    sdf_writer.write(mol_block + '\n$$$$\n')
                    num_mol += 1
    
    def run_file_list(self, sdf_list, save_name='Filter_Result.sdf'):
        num_mol = 0
        smi_dict = {}
        for sdf in sdf_list:
            supp = Chem.SDMolSupplier(sdf)
            for mol in supp:
                if mol is None: continue
                mol_ = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
                out = self._filter(mol_)
                if out != False and out[0] not in smi_dict:
                    mol_name = 'No_{}-{}'.format(num_mol, sdf)
                    smi_dict.setdefault(out[0])
                    with open(save_name, 'a') as sdf_writer:
                        #mol = out[1]
                        mol.SetProp('_Name', mol_name)
                        mol_block = Chem.MolToMolBlock(mol)
                        sdf_writer.write(mol_block + '\n$$$$\n')
                        num_mol += 1
        return num_mol
    
    def np_run(self, mol_list, save_name='Filter_Result.sdf', n_process=10):
        smi_dict = {}
        pool = Pool(processes=n_process)
        filter_list = pool.map(self._filter, mol_list)
        #filter(lambda x:x!=False, filter_list)
        for ix, i in enumerate(filter_list):
            if i:
                if i[0] not in smi_dict:
                    smi_dict.setdefault(i[0])
                    with open(save_name, 'a') as sdf_writer:
                        sdf_writer.write(i[1] + '\n$$$$\n')

                        

from PIL import Image
import matplotlib.pyplot as plt
from .train import verify_dir_exists
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
import os


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
        legend = mw + '\n' + docking_score
        legends.append(legend)

    img = Draw.MolsToGridImage(
        mol_list,
        molsPerRow=molsPerRow,
        subImgSize=subImgSize,
        legends=legends
    )
    return img


def DrawDockedMol(mol,         # rdkit.Molecule的instance
            save=None, 
            size=(300,300), 
            bondLineWidth=5,   # 化学键显示的宽度
            FontSize=3,
            fixedBondLength=50,    # 化学键显示的长度
            legend=None,
            legendFontSize=5,
            elemColor=True):  # 是否按元素给原子着色
    
    if legend is None:
        try:
            docking_score = 'Docking Score: {:.3f}'.format(float(mol.GetProp('r_i_docking_score')))
            mw = 'MolWt: {:.3f}'.format(Descriptors.MolWt(mol))
            legend = mw + '\n' + docking_score
        except:
            pass
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
    AllChem.Compute2DCoords(mol)
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
    

def CombineImages(img_file_list, col_num=7, save_dir='./', title='merged_image.png', img_size=None):
    num_img = len(img_file_list)
    num_row = num_img//col_num+1 if num_img%col_num != 0 else num_img//col_num
    if img_size is None:
        img_size = Image.open(img_file_list[0]).size
    toImage = Image.new('RGB', (img_size[1]*col_num, img_size[0]*num_row), color=(255,255,255))
    x_cusum = 0
    y_cusum = 0
    num_has_paste = 0
    for img_file in img_file_list:
        img = Image.open(img_file)
        toImage.paste(img, (x_cusum, y_cusum))
        num_has_paste += 1
        if num_has_paste%col_num==0:
            x_cusum = 0
            y_cusum += img_size[1]
        else:
            x_cusum += img_size[0]
    verify_dir_exists(save_dir)
    toImage.save(save_dir+'/'+title)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')  # 取消所有边框，坐标轴
    plt.imshow(toImage)
    plt.clf()    # 清除当前图形。如果不清除，当使用循环大量作图，机器会为plt分配越来越多的内存，速度会逐渐变慢。