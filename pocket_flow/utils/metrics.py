from rdkit import Chem



FUSED_QUA_RING_PATTERN = [
    Chem.MolFromSmarts(i) for i in[
        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R](~&@[R]~&@1~&@4)~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R]~&@3~&@[R](~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]2~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]2~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]34~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@4',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@1)~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@4~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R@@H](~&@[R@H]4~&@[R]~&@[R]~&@[R]~&@[R]~&@[R@H]~&@4~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]1~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@2',
         '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@[R]~&@[R]~&@3',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]~&@1)~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@[R]4~&@[R]~&@[R]~&@[R]~&@[R]~&@4~&@[R]~&@3)~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@[R]~&@4)~&@[R]~&@3~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]3~&@[R]4~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@4~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@13~&@[R](~&@[R]~&@[R]~&@[R]~&@3)~&@[R]~&@[R]1~&@[R]~&@2~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R]~&@2~&@[R]2~&@[R]~&@[R]~&@[R]~&@[R]~&@2~&@[R]~&@[R]~&@1'
            ]
        ]

def has_fused4ring(mol):
    for pat in FUSED_QUA_RING_PATTERN:
        if mol.HasSubstructMatch(pat):
            return True
    else:
        return False


PATTERNS_1 = [Chem.MolFromSmarts(i) for i in [
                        '[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@1~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
                         '[R]12~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@2',
                         '[R]1~&@[R]~&@[R]~&@12~&@[R]~&@[R]~&@2'
                        ]
           ]

def judge_unexpected_ring(mol):
    for pat in PATTERNS_1:
        subs = mol.GetSubstructMatches(pat)
        if len(subs) > 0:
            return True
    else:
        return False


PATTERNS = [Chem.MolFromSmarts(i) for i in [
                        '[R]12~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2',
                        '[R]12~&@[R]~&@[R]3~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@3~&@[R]~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@2',
                        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]3~&@[R]~&@1~&@[R](~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
                        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1',
                        '[R]12~&@[R]~&@[R]~&@[R]3~&@[R](~&@[R]~&@1~&@[R]~&@[R]~&@[R]~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@3',
                        '[R]12~&@[R]~&@[R]~&@[R]~&@[R]~&@[R]~&@1~&@[R]~&@[R]1~&@[R](~&@[R]~&@2)~&@[R]~&@[R]~&@[R]~&@[R]~&@1']
           ]

def judge_fused_ring(mol):
    for pat in PATTERNS+FUSED_QUA_RING_PATTERN:
        if mol.HasSubstructMatch(pat):
            return True
    else:
        return False

def substructure(mol_lib):
    total_num = sum([len(i) for i in mol_lib])
    ring_size_statis = {
            'tri_ring':{'num':0},
            'qua_ring':{'num':0},
            'fif_ring':{'num':0},
            'hex_ring':{'num':0},
            'hep_ring':{'num':0},
            'oct_ring':{'num':0},
            'big_ring':{'num':0},
            'fused_ring':{'num':0},
            'unexpected_ring':{'num':0},
            'sssr':{}
                       }
    for s in mol_lib:
        for mol in s:
            if mol is None: continue
            mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=False))
            sssr = Chem.GetSSSR(mol)
            if sssr in ring_size_statis['sssr']:
                ring_size_statis['sssr'][sssr]['num'] += 1
            else:
                ring_size_statis['sssr'][sssr] = {}
                ring_size_statis['sssr'][sssr]['num'] = 1
            ring = mol.GetRingInfo()
            has_ring_size_3 = False
            has_ring_size_4 = False
            has_ring_size_5 = False
            has_ring_size_6 = False
            has_ring_size_7 = False
            has_ring_size_8 = False
            has_big_ring = False
            has_fused_ring = False
            has_unexpected_ring = False
            for r in mol.GetRingInfo().AtomRings():
                if len(r) == 3 and has_ring_size_3 is False:
                    has_ring_size_3 = True
                    ring_size_statis['tri_ring']['num'] += 1
                if len(r) == 4 and has_ring_size_4 is False:
                    has_ring_size_4 = True
                    ring_size_statis['qua_ring']['num'] += 1
                if len(r) == 5 and has_ring_size_5 is False:
                    has_ring_size_5 = True
                    ring_size_statis['fif_ring']['num'] += 1
                if len(r) == 6 and has_ring_size_6 is False:
                    has_ring_size_6 = True
                    ring_size_statis['hex_ring']['num'] += 1
                if len(r) == 7 and has_ring_size_7 is False:
                    has_ring_size_7 = True
                    ring_size_statis['hep_ring']['num'] += 1
                if len(r) == 8 and has_ring_size_8 is False:
                    has_ring_size_8 = True
                    ring_size_statis['oct_ring']['num'] += 1
                if len(r) > 8 and has_big_ring is False:
                    has_big_ring = True
                    ring_size_statis['big_ring']['num'] += 1
            if judge_fused_ring(mol) and has_fused_ring is False:
                has_fused_ring = True
                ring_size_statis['fused_ring']['num'] += 1
            if judge_unexpected_ring(mol) and has_unexpected_ring is False:
                ring_size_statis['unexpected_ring']['num'] += 1
    ring_size_statis['tri_ring']['rate'] = ring_size_statis['tri_ring']['num']/total_num
    ring_size_statis['qua_ring']['rate'] = ring_size_statis['qua_ring']['num']/total_num
    ring_size_statis['fif_ring']['rate'] = ring_size_statis['fif_ring']['num']/total_num
    ring_size_statis['hex_ring']['rate'] = ring_size_statis['hex_ring']['num']/total_num
    ring_size_statis['hep_ring']['rate'] = ring_size_statis['hep_ring']['num']/total_num
    ring_size_statis['oct_ring']['rate'] = ring_size_statis['oct_ring']['num']/total_num
    ring_size_statis['big_ring']['rate'] = ring_size_statis['big_ring']['num']/total_num
    ring_size_statis['fused_ring']['rate'] = ring_size_statis['fused_ring']['num']/total_num
    ring_size_statis['unexpected_ring']['rate'] = ring_size_statis['unexpected_ring']['num']/total_num
    for k in ring_size_statis['sssr']:
        ring_size_statis['sssr'][k]['rate'] = ring_size_statis['sssr'][k]['num']/total_num
    return ring_size_statis


def smoothing(scalars, weight=0.8):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed