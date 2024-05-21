import argparse
from pathlib import Path

from pocket_flow import Ligand, Protein, SplitPocket


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, required=True, help="PDB")
    parser.add_argument("--ligand", type=str, required=True, help="SDF")
    args = parser.parse_args()

    return args


def main():

    args = arguments()

    output_path = f"{Path(args.protein).parent}/pocket.pdb"

    pro = Protein(args.protein)
    lig = Ligand(args.ligand)
    dist_cutoff = 10
    pocket_block, _ = SplitPocket._split_pocket_with_surface_atoms(
        pro, lig, dist_cutoff
    )
    open(output_path, "w").write(pocket_block)


if __name__ == "__main__":
    main()
