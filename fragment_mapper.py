import csv
import signal
from rdkit import Chem
from rdkit.Chem import rdqueries

from collections import defaultdict

from copy import deepcopy

from rdkit_util import MolToIsoSmiles

class Piece(object):
    def __init__(self, smi, matched_indices):
        self.smi = smi
        self.matched_indices = set(matched_indices)

    def get_num_atoms(self):
        return len(self.matched_indices)

    def get_canonical_smiles(self):
        return self.smi

    def get_matched_indices(self):
        return self.matched_indices

class MatchedPieces(object):
    def __init__(self, target):
        self.atom_indices_left = set(range(target.GetNumAtoms()))
        self.pieces = []

    def valid_piece_to_add(self, piece):
        return self.atom_indices_left.issuperset(piece.matched_indices)

    def get_remaining_possible(self, possible_pieces):
        indices_left = set(self.atom_indices_left)

        new_possible_pieces = []
        for piece in possible_pieces:
            if self.valid_piece_to_add(piece):
                new_possible_pieces.append(piece)
                indices_left -= piece.get_matched_indices()

        if indices_left:
            raise NotFullyCovered(indices_left)

        return new_possible_pieces

    def add_piece(self, piece):
        assert len(self.atom_indices_left & piece.matched_indices) == len(piece.matched_indices)

        self.atom_indices_left -= piece.matched_indices
        self.pieces.append(piece)

    def is_complete(self):
        return len(self.atom_indices_left) == 0

    def get_all_mols(self):
        return [Chem.MolFromSmiles(p.smi) for p in self.pieces]

    def get_all_smiles(self):
        return '.'.join(p.smi for p in self.pieces)

    def get_hashable(self):
        d = defaultdict(lambda : 0)
        for p in self.pieces:
            d[p.smi] += 1

        return tuple(sorted(d.items()))

    def __eq__(self, other):
        return self.get_hashable() == other.get_hashable()

    def __hash__(self):
        return hash(self.get_hashable())

def recursive_find_matches(matched_pieces, possible_pieces, matched_piece):
    #origsmi = matched_pieces.get_all_smiles()
    matched_pieces = deepcopy(matched_pieces)
    matched_pieces.add_piece(matched_piece)
    #print("%s + %s = %s" % (origsmi, matched_piece.smi, matched_pieces.get_all_smiles()))
    if matched_pieces.is_complete():
        yield matched_pieces
        return

    try:
        remaining_possible = matched_pieces.get_remaining_possible(possible_pieces)
    except NotFullyCovered as e: # not possible to finish, to terminate searching here
        #print("Not possible to finish, terminating this search path")
        return

    for idx, next_piece in enumerate(remaining_possible):
        for match in recursive_find_matches(matched_pieces, remaining_possible[idx + 1:], next_piece):
            yield match

def enumerate_all_matches(target, possible_pieces):
    matched_pieces = MatchedPieces(target)

    # this throws NotFullyCovered back up to the caller if it's never possible
    remaining_possible = matched_pieces.get_remaining_possible(possible_pieces)

    for idx, next_piece in enumerate(remaining_possible):
        for match in recursive_find_matches(matched_pieces, remaining_possible[idx + 1:], next_piece):
            yield match

class NotFullyCovered(Exception):
    def __init__(self, missing_indices):
        self.missing_indices = missing_indices

class MappingNotFound(Exception):
    pass

class TimeLimitExceeded(Exception):
    pass

def make_fragment_query(smi):
    mol = Chem.RWMol(Chem.MolFromSmarts(smi))
    for atom in mol.GetAtoms():
        newatom = rdqueries.AtomNumEqualsQueryAtom(atom.GetAtomicNum())
        newatom.ExpandQuery(rdqueries.IsAromaticQueryAtom(not atom.GetIsAromatic()))
        newatom.ExpandQuery(rdqueries.IsInRingQueryAtom(not atom.IsInRing()))

        if atom.GetIsotope():
            newatom.ExpandQuery(rdqueries.IsotopeEqualsQueryAtom(atom.GetIsotope()))

        mol.ReplaceAtom(atom.GetIdx(), newatom)

    return mol

class FragmentMapper(object):
    def __init__(self, fragments_fname=None):
        self.fragments = {}

        if fragments_fname is not None:
            reader = csv.DictReader(open(fragments_fname))

            for row in reader:
                self.add_fragment(row["SMILES"])

    def add_fragment(self, fragment_smiles):
        orig_mol = Chem.MolFromSmiles(fragment_smiles)
        cansmi = MolToIsoSmiles(orig_mol)

        mol = make_fragment_query(cansmi)
        if cansmi not in self.fragments:
            self.fragments[cansmi] = mol

    def get_possible_matches(self, target):
        possible_pieces = []
        for smi, frag in self.fragments.items():
            matches = target.GetSubstructMatches(frag)

            for matched_indices in matches:
                possible_pieces.append(Piece(smi, matched_indices))

        # try the large matches first
        possible_pieces.sort(key=Piece.get_num_atoms, reverse=True)
        return possible_pieces

    def _match_fragments(self, target):
        possible_pieces = self.get_possible_matches(target)

        seen = set()
        for completed_match in enumerate_all_matches(target, possible_pieces):
            if completed_match in seen:
                continue

            seen.add(completed_match)
            yield completed_match

    def get_best_match(self, target, timelimit=10):
        def alarm_handler(signal, frame):
            raise TimeLimitExceeded

        old_alarm = signal.signal(signal.SIGALRM, alarm_handler)
        try:
            signal.alarm(timelimit)
            for match in self._match_fragments(target):
                return match

        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_alarm)

        raise MappingNotFound()

