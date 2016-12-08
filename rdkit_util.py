from __future__ import print_function
import os
import sys
import re
import time
import csv
import gzip
from StringIO import StringIO

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import Font, DrawingOptions
from rdkit.Chem.Draw.spingCanvas import Canvas
from rdkit.sping import pagesizes

def ECFP4Fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def MolToIsoSmiles(m):
    isomericSmiles = True
    return Chem.MolToSmiles(m, isomericSmiles)

dialect = {
    "delimiter" : "|",
    "quotechar" : "'",
    "lineterminator" : ""
}

def escape_multiline_field(field):
    lines = field.splitlines()
    if len(lines) < 2:
        return field

    fobj = StringIO()
    writer = csv.writer(fobj, **dialect)
    writer.writerow(lines)
    return fobj.getvalue()

def unescape_multiline_field(field):
    if not field:
        return field

    fobj = StringIO(field)
    reader = csv.reader(fobj, **dialect)
    for row in reader:
        return "\n".join(row)

    raise ValueError("Nothing found, something is seriously wrong!")

def get_smiles_from_csv_line(line):
    for row in csv.reader(StringIO(line)):
        return row[0]
    raise ValueError("No CSV lines found!")

class CSVWriter(Chem.SmilesWriter):
    def __init__(self, *args, **kwargs):
        kwargs["delimiter"] = ","
        if "isomericSmiles" not in kwargs:
            kwargs["isomericSmiles"] = True
        Chem.SmilesWriter.__init__(self, *args, **kwargs)

        self._props = None

    def GetProps(self):
        return self._props

    def write(self, mol, *args, **kwargs):
        Chem.AssignAtomChiralTagsFromStructure(mol)
        if self._props is None:
            self._props = list(mol.GetPropNames())
            self.SetProps(self._props)

        for propname in self._props:
            try:
                current_value = mol.GetProp(propname)
                mol.SetProp(propname, escape_multiline_field(current_value))
            except KeyError:
                pass

        Chem.SmilesWriter.write(self, mol, *args, **kwargs)

class CSVMolSupplier(Chem.SmilesMolSupplier):
    def __init__(self, *args, **kwargs):
        kwargs["delimiter"] = ","
        Chem.SmilesMolSupplier.__init__(self, *args, **kwargs)

    def next(self):
        mol = Chem.SmilesMolSupplier.next(self)

        for name in mol.GetPropNames():
            new_field = unescape_multiline_field(mol.GetProp(name))
            mol.SetProp(name, new_field)

        return mol

    def __iter__(self):
        return self

def GetMolSupplierFromFilename(fname):
    base, ext = os.path.splitext(fname)
    if ext == ".smi":
        return Chem.SmilesMolSupplier(fname)

    if ext == ".gz":
        base, ext = os.path.splitext(base)
        if ext == ".sdf":
            return Chem.ForwardSDMolSupplier(gzip.open(fname))

    if ext == ".csv":
        return CSVMolSupplier(fname)

    return Chem.SupplierFromFilename(fname)

def GetWriterFromFilename(fname):
    base, ext = os.path.splitext(fname)
    if ext == ".csv":
        return CSVWriter(fname)

    if ext == ".sdf":
        return Chem.SDWriter(fname)

    raise ValueError("Unrecognized file extension: %s", ext)

class Timer(object):
    def __init__(self):
        self.bgn = time.time()
    def Elapsed(self):
        return time.time() - self.bgn

class MolRowReport(object):
    def __init__(self, ofname, numRows=8, percentMainMol=0.2, mainTitle=None, otherTitle=None, description=None):
        self.numRows = numRows
        self.percentMainMol = percentMainMol

        self.xdim, self.ydim = pagesizes.A4

        self.canvas = Canvas((self.xdim, self.ydim), ofname, imageType="pdf")

        self.xmargin = self.xdim * 0.2
        self.ymargin = self.ydim * 0.1

        remainingXSize = (self.xdim - self.xmargin)
        self.mainMolSize = remainingXSize * self.percentMainMol
        self.rowSize = remainingXSize - self.mainMolSize
        self.ysize = (self.ydim - self.ymargin) / numRows

        self.current_row_idx = 0
        self.num_pages = 0

        if description is not None:
            font = self._get_font()
            ypos = self.ymargin * 0.375
            xpos = (self.xmargin +
                    self._correct_for_rdkit_going_off_into_the_wild_blue_yonder(description, (self.xmargin, ypos, 0), font)
                    + 100)

            self.canvas.addCanvasText(description, (xpos, ypos, 0), font)
            self._new_page()

        self.mainTitle = mainTitle
        self.otherTitle = otherTitle
        self._write_header()

    def _correct_for_rdkit_going_off_into_the_wild_blue_yonder(self, text, pos, font):
        """
        Copy and pasted from rdkit/Chem/Draw/spingCanvas.py because
        with large strings this places the text way off the canvas.
        """
        from rdkit.sping import pid
        from rdkit.Chem.Draw.spingCanvas import faceMap
        text = re.sub(r'\<.+?\>', '', text)
        font = pid.Font(face=faceMap[font.face], size=font.size)
        txtWidth, txtHeight = self.canvas.canvas.stringBox(text, font)
        bw, bh = txtWidth + txtHeight * 0.4, txtHeight * 1.4
        offset = txtWidth * pos[2]
        correction = -(pos[0] - txtWidth / 2 + offset)
        return correction

    def _new_page(self):
        self.canvas.canvas.showPage()

    def _get_font(self):
        return Font(face='sans', size=8)

    def _write_header(self):
        ypos = self.ymargin * 0.375
        font = self._get_font()
        if self.mainTitle is not None:
            xpos = self.xmargin
            self.canvas.addCanvasText(self.mainTitle, (xpos, ypos, 0), font)

        if self.otherTitle is not None:
            xpos = self.xmargin + self.mainMolSize
            self.canvas.addCanvasText(self.otherTitle, (xpos, ypos, 0), font)

    def _mol_to_image(self, mol, size, trans, highlightAtoms=None, coordScale=None, **kwargs):
        options = DrawingOptions()
        options.bgColor = None
        options.radicalSymbol = '.' # sping doesn't handle unicode
        options.atomLabelFontSize = 4
        options.atomLabelMinFontSize = 4

        if highlightAtoms is not None:
            options.elemDict = {}

        if coordScale is not None:
            options.coordScale = coordScale

        color = (1.0, 0.0, 0.0)

        Draw.MolToImage(mol,
                        size=size,
                        fitImage=True,
                        canvas=self.canvas,
                        options=options,
                        centerIt=False,
                        drawingTrans=trans,
                        highlightAtoms=highlightAtoms,
                        highlightColor=color,
                        **kwargs)

    def _get_max_per_row(self):
        return 6

    def _get_rows_and_columns(self, nitems):
        max_per_row = self._get_max_per_row()
        nrows = ((nitems - 1) / max_per_row) + 1
        ncols = nitems / nrows
        if nitems > max_per_row and (nitems % nrows) != 0:
            ncols += 1
        return nrows, ncols

    def _get_coord_scale(self, nitems):
        nrows, ncols = self._get_rows_and_columns(nitems)
        if nrows > 1:
            return 2.0
        return 1.0

    def add_row(self, main_mol, other_mols, highlightAtoms=None, **kwargs):
        curYPos = self.current_row_idx

        if self.current_row_idx == 0:
            if self.num_pages != 0:
                self._new_page()
                self._write_header()
            self.num_pages += 1

        xtrans = self.xmargin
        ytrans = self.ydim - (self.ysize * (curYPos + 1))
        trans = (xtrans, ytrans)
        size = (self.mainMolSize, self.ysize)
        self._mol_to_image(main_mol, size, trans, highlightAtoms=highlightAtoms, **kwargs)

        labelYPos = (self.ysize * curYPos) + (self.ysize / 2.0)
        if main_mol.HasProp("_Name"):
            labelPos = (xtrans, labelYPos, 0)
            self.canvas.addCanvasText(main_mol.GetProp("_Name"), labelPos, self._get_font())

        # turns off colored atoms in _mol_to_image
        if highlightAtoms is not None:
            highlightAtoms = []

        num_mols = len(other_mols)
        nrows, ncols = self._get_rows_and_columns(num_mols)

        xsize = self.rowSize / float(ncols)
        ysize = self.ysize / float(nrows)
        xbase = self.xmargin + self.mainMolSize

        ybase = ytrans + (self.ysize / 2.0)

        coordScale = self._get_coord_scale(num_mols)

        for molidx, mol in enumerate(other_mols):
            rowidx = molidx / ncols
            colidx = molidx % ncols

            xtrans = xbase + xsize * colidx
            ytrans = ybase - ysize * rowidx - (ysize / 2.0)

            trans = (xtrans, ytrans)
            size = (xsize, ysize)

            self._mol_to_image(mol, size, trans, highlightAtoms=highlightAtoms, coordScale=coordScale, **kwargs)

        lineYPos = labelYPos - 8
        self.canvas.addCanvasLine((self.xmargin / 2, lineYPos), (self.xdim - (self.xmargin / 2), lineYPos))

        curYPos += 1
        if curYPos == self.numRows:
            curYPos = 0

        self.current_row_idx = curYPos

    def __del__(self):
        self.canvas.save()

class MolGridReport(object):
    def __init__(self, ofname, numCols, numRows):
        self.numCols = numCols
        self.numRows = numRows

        self.xdim, self.ydim = pagesizes.A4

        self.canvas = Canvas((self.xdim, self.ydim), ofname, imageType="pdf")

        self.xmargin = self.xdim * 0.2
        self.ymargin = self.ydim * 0.2

        self.xsize = (self.xdim - self.xmargin) / numCols
        self.ysize = (self.ydim - self.ymargin) / numRows

        self.current_mol_idx = (0, 0)
        self.num_pages = 0

    def add_mol(self, mol, **kwargs):
        curXPos, curYPos = self.current_mol_idx

        if self.current_mol_idx == (0, 0):
            if self.num_pages != 0:
                self.canvas.canvas.showPage()
            self.num_pages += 1

        options = DrawingOptions()
        options.bgColor = None
        options.radicalSymbol = '.' # sping doesn't handle unicode
        options.atomLabelFontSize = 4
        options.atomLabelMinFontSize = 4

        xtrans = self.xmargin + (self.xsize * curXPos)
        ytrans = self.ydim - (self.ysize * (curYPos + 1))
        trans = (xtrans, ytrans)
        Draw.MolToImage(mol,
                        size=(self.xsize, self.ysize),
                        fitImage=True,
                        canvas=self.canvas,
                        options=options,
                        centerIt=False,
                        drawingTrans=trans,
                        **kwargs)

        if mol.HasProp("_Name"):
            labelYPos = (self.ysize * curYPos) + (self.ysize / 2.0)
            labelPos = (xtrans, labelYPos, 0)
            font=Font(face='sans', size=8)
            self.canvas.addCanvasText(mol.GetProp("_Name"), labelPos, font)

        curXPos += 1
        if curXPos == self.numCols:
            curXPos = 0
            curYPos += 1
            if curYPos == self.numRows:
                curYPos = 0

        self.current_mol_idx = (curXPos, curYPos)

    def __del__(self):
        self.canvas.save()


def main(argv=[__name__]):
    if len(argv) != 3:
        print("%s <first smiles> <second smiles>" % argv[0])
        print("Be sure to escape the SMILES properly on the command line!")
        return 1

    mol1 = Chem.MolFromSmiles(argv[1])
    fp1 = ECFP4Fingerprint(mol1)

    mol2 = Chem.MolFromSmiles(argv[2])
    fp2 = ECFP4Fingerprint(mol2)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    print("Similarity between molecules =", similarity)

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
