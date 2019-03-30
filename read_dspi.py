import pandas as pd
from rdkit.Chem import Descriptors
import rdkit.Chem as Chem

def Nrot(row):
    """ Get number of rotatble bonds

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    N_rot: int
        Number of rotatable bonds
    """
    m = Chem.MolFromSmiles(row.SMILES)
    N_rot = Descriptors.NumRotatableBonds(m)

    return N_rot


def heavy_atoms(row):
    """ Get number of Heavy atoms

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    heavy_atom_count: int
        Number of heavy atoms
    """
    m = Chem.MolFromSmiles(row.SMILES)
    heavy_atom_count = Descriptors.HeavyAtomCount(m)

    return heavy_atom_count


def clogP(row):
    """ Get number of Heavy atoms

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    cLogP: float
        Calculated partition coefficient of n-octonal and water
    """
    m = Chem.MolFromSmiles(row.SMILES)
    clogp = Descriptors.MolLogP(m)

    return clogp


def TPSA(row):
    """Get Total polar surface area

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    total_polar_surface_area: float
        total polar surface area
    """
    m = Chem.MolFromSmiles(row.SMILES)
    total_polar_surface_area = Descriptors.TPSA(m)
    return total_polar_surface_area


def NDon(row):
    """Get Number of H-Bond Donors

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    donors: int
        Number of Donors
    """
    m = Chem.MolFromSmiles(row.SMILES)
    donors = Descriptors.NumHDonors(m)
    return donors

def NDon(row):
    """Get Number of H-Bond Donors

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    donors: int
        Number of Donors
    """
    m = Chem.MolFromSmiles(row.SMILES)
    donors = Descriptors.NumHDonors(m)
    return donors


def NAcc(row):
    """Get Number of H-Bond Acceptors

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    Acceptors: int
        Number of Acceptors
    """
    m = Chem.MolFromSmiles(row.SMILES)
    acceptors = Descriptors.NumHAcceptors(m)
    return acceptors


def Fsp3(row):
    """Get Fraction of carbons that are sp3

    Parameters
    ----------
    row:
        row of pandas.DataFrame containing SMILES field

    Returns
    -------
    FSP3:
        Fraction of carbons that are sp3
    """
    m = Chem.MolFromSmiles(row.SMILES)
    FSP3 = Descriptors.FractionCSP3(m)
    return FSP3


DSPI_df = pd.read_excel('DSPI.xlsx')

DSPI_df['N_rot'] = DSPI_df.apply(Nrot, axis=1)
DSPI_df['HAC'] = DSPI_df.apply(heavy_atoms, axis=1)
DSPI_df['cLogP'] = DSPI_df.apply(clogP, axis =1)
DSPI_df['TSPA'] = DSPI_df.apply(TPSA, axis=1)
DSPI_df['NDon'] = DSPI_df.apply(NDon, axis=1)
DSPI_df['NAcc'] = DSPI_df.apply(NAcc, axis=1)
DSPI_df['Fsp3'] = DSPI_df.apply(Fsp3, axis=1)

print(DSPI_df.mean())
print(DSPI_df.SMILES.nunique())