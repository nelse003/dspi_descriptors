import pandas as pd
from rdkit.Chem import Descriptors
import rdkit.Chem as Chem
import matplotlib.pyplot as plt

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

def add_properties_to_df(df):

    """ Add predicted properties to DataFrame with SMILES

    Adds new column for each of:

    N_rot: Number of rotatable bonds
    HAC: Heavy atom count
    cLogP: Calculated log partition coefficent
    TSPA: Total polar surface area
    NDON: number of Hbond donors
    NAcc: Number of Hbond Acceptors
    Fsp3: Fraction of sp3 carbons
    """

    df['N_rot'] = df.apply(Nrot, axis=1)
    df['HAC'] = df.apply(heavy_atoms, axis=1)
    df['cLogP'] = df.apply(clogP, axis =1)
    df['TSPA'] = df.apply(TPSA, axis=1)
    df['NDon'] = df.apply(NDon, axis=1)
    df['NAcc'] = df.apply(NAcc, axis=1)
    df['Fsp3'] = df.apply(Fsp3, axis=1)

    return df

def plot_histograms(df, library_name):

    properties = ['N_rot', 'HAC', 'cLogP', 'TSPA', 'NDon', 'NAcc', 'Fsp3']

    for prop in properties:
        df.plot(y=prop, kind='hist', bins=40)
        plt.xlabel('{}'.format(prop))
        plt.title('{}'.format(library_name))
        plt.savefig('{}_{}.png'.format(prop, library_name), dpi=300)


if __name__ == '__main__':

    dspi_df = pd.read_excel('DSPI.xlsx')
    dspi_df = add_properties_to_df(dspi_df)

    print("DPSI")
    print(dspi_df.mean())
    print(dspi_df.SMILES.nunique())
    print("_-------------------------")

    plot_histograms(dspi_df,"DSPI")
    dspi_df.to_excel("DSPI_with_properties.xlsx")

    minifrag_df = pd.read_excel('MiniFrag.xlsx')
    minifrag_df = add_properties_to_df(minifrag_df)

    print("MiniFrag")
    print(minifrag_df.mean())
    print(minifrag_df.SMILES.nunique())
    print("_-------------------------")

    plot_histograms(minifrag_df,"MiniFrag")
    minifrag_df.to_excel("minifrag_with_properties.xlsx")

