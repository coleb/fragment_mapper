#! /usr/bin/env python
from __future__ import print_function
import os
import sys
import argparse
import csv
from rdkit import Chem

TOP = os.path.join(os.path.dirname(__file__))
sys.path.append(TOP)

from rdkit_util import (
    GetMolSupplierFromFilename,
    Timer,
    MolRowReport,
    GetWriterFromFilename)

from fragment_mapper import (FragmentMapper,
                             NotFullyCovered,
                             MappingNotFound,
                             TimeLimitExceeded)

def get_unique_mols(pieces):
    mols = []
    seen = set()
    for p in pieces:
        cansmi = p.get_canonical_smiles()
        if cansmi in seen:
            continue
        seen.add(cansmi)
        mols.append(Chem.MolFromSmiles(cansmi))
    return mols

SUCCESS_DESCRIPTION = """

MOLECULES THAT CAN BE SUCCESSFULLY CONSTRUCTED FROM THE GIVEN FRAGMENTS

The drug molecules on the left can be successfully constructed by
combining the fragments on the right. The fragments on the right are
the smallest set of fragments that can achieve that from the given
fragments. Therefore, instead of combining N carbons together to make
a long chain of carbons, the longer alkyl chain fragments are
preferred. The fragments on the right aren't the 'only' way to
construct the molecule from the given fragments, just the first
smallest set found (there could be other sets of fragments of equal
size as well).

*** WARNING ***

Bond orders are _not_ taken into account because I just haven't got
around to coding in the check to make sure the bond orders are the
same. For example, a hydroxyl attached to a ring system is not
differentiated from a double-bonded exocyclic carbonyl. Maybe it could
be argued that this could handled by tautomer enumeration instead?
Thoughts?
"""

MAPPING_NOT_FOUND_DESCRIPTION = """

MOLECULES THAT CAN'T BE CONSTRUCTED FROM THE GIVEN FRAGMENTS

These are molecules that are completely covered in some way by the
fragments given. However, there is no way to connect the fragments on
the right through simple solitary bonds to build up the larger
drug molecule on the left.

For example, if benzene and naphthalene are in the fragment library
and anthracene is in the drug molecule on the left then the molecule
will be in this pdf. This is because benzene and naphthalene are both
contained in anthracene, but they share the same atoms in order to
construct anthracene.

All fragments that are substructures of the drug molecule on the left
are listed on the right.

"""

NOT_FULLY_COVERED_DESCRIPTION = """

MOLECULES THAT CONTAIN SUBSTRUCTURES NOT PRESENT IN THE GIVEN FRAGMENTS

These are molecules that contain atom topologies not present in the
fragment library given. The atoms highlighted in red are the atom
topologies not present.

All fragments that are substructures of the drug molecule on the left
are listed on the right that are contained.

"""

class MoleculeFeasibility(object):
    def __init__(self, args):
        self.args = args
        self.mapper = FragmentMapper(args.fragments)
        self.success_report = MolRowReport("Success.pdf", 8, mainTitle="Drug",
                                           otherTitle="Smallest set of matching fragments",
                                           description=SUCCESS_DESCRIPTION)
        self.mapping_not_found = MolRowReport("MappingNotFound.pdf", 8, mainTitle="Drug",
                                              otherTitle="All matching fragments",
                                              description=MAPPING_NOT_FOUND_DESCRIPTION)
        self.not_fully_covered = MolRowReport("NotFullyCovered.pdf", 8, mainTitle="Drug",
                                              otherTitle="All matching fragments",
                                              description=NOT_FULLY_COVERED_DESCRIPTION)
        self.time_limit_exceeded = MolRowReport("TimeLimitExceeded.pdf", 8, mainTitle="Drug", otherTitle="All matching fragments")

    def check_mol(self, target):
        if isinstance(target, str):
            target = Chem.MolFromSmiles(target)
            assert target is not None
        desc = target.GetProp("_Name") if target.HasProp("_Name") else Chem.MolToSmiles(target)
        print("Matched substructures to %s:" % desc)
        count = 0
        pieces = self.mapper.get_possible_matches(target)
        for piece in pieces:
            print(piece.get_canonical_smiles(), piece.get_matched_indices())
            count += 1
        print("%i matched substructures" % count)

        target.SetProp("NUM_MATCHES", str(count))
        target.SetProp('MATCHED_SMILES', '')

        timer = Timer()
        try:
            matched = self.mapper.get_best_match(target, self.args.timelimit)
            all_smiles = matched.get_all_smiles()
            print("Found", all_smiles, "in %.3f seconds" % timer.Elapsed())
            target.SetProp('MATCHED_SMILES', all_smiles)
            target.SetProp('REASON', 'SUCCESS')

            self.success_report.add_row(target, matched.get_all_mols())
        except NotFullyCovered as e:
            print("The following indices in the drug are simply not matched: %r" % e.missing_indices)
            target.SetProp('REASON', "NotFullyCovered")

            self.not_fully_covered.add_row(target, get_unique_mols(pieces), highlightAtoms=e.missing_indices)

        except MappingNotFound as e:
            print("Not possible to find a mapping after %.3f seconds" % timer.Elapsed())
            target.SetProp('REASON', "MappingNotFound")

            self.mapping_not_found.add_row(target, get_unique_mols(pieces))

        except TimeLimitExceeded as e:
            print("Time limit for search exceeded after %.3f seconds" % timer.Elapsed())
            target.SetProp('REASON', "TimeLimitExceeded")

            self.time_limit_exceeded.add_row(target, get_unique_mols(pieces))

        target.SetProp('ELAPSED_TIME', "%.3f" % timer.Elapsed())
        return target

def main(argv=[__name__]):
    parser = argparse.ArgumentParser(description='determine whether it is possible to reach a particular molecule through the given fragment library')
    parser.add_argument('-f', '--fragments',
                        type=str,
                        default=os.path.join(TOP, "fragments.csv"),
                        help="fragment database to use")
    parser.add_argument('-t', '--timelimit',
                        type=int,
                        default=600,
                        help="Maximum amount of time to allow for mapping a molecule")

    parser.add_argument('smiles', type=str, nargs=1, help="The SMILES of the molecule to figure out if it is covered by this fragment library through single bonds")
    parser.add_argument('output', type=str, nargs=1, help="The CSV file containing information about the mapping")
    args = parser.parse_args(argv[1:])

    feasible = MoleculeFeasibility(args)

    smiles = args.smiles[0]
    if not os.path.exists(smiles):
        feasible.check_mol(smiles)
        return 0

    reader = GetMolSupplierFromFilename(smiles)
    writer = GetWriterFromFilename(args.output[0])

    for molidx, mol in enumerate(reader):
        #if molidx >= 50:
        #    break

        mol = feasible.check_mol(mol)
        writer.write(mol)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
