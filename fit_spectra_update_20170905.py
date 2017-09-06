#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# parse arguments provided by command line user
parser = argparse.ArgumentParser()
parser.add_argument("-file", default='merged_spectra.txt', type=str,
                    help="""name of file containing sample and reference
                    spectra""")
parser.add_argument("-nuc", default='GDP', type=str,
                    help="""name of nucleotide spectrum to be subtracted""")
parser.add_argument("-output", default="fit_spectra.png", type=str,
                    help="""output figure filename""")

args = parser.parse_args()
filename = args.file
nucleotide = args.nuc
output = args.output

# read spectra file into pandas dataframe
df = pd.read_csv(filename, sep='\t', header=0)

# extract vectors for sample, protein reference, and nucleotide reference
# spectra, as  well as wavelength
pn = np.array(df[df['sample'] == 'protein'].Abs)
p = np.array(df[df['sample'] == 'unfolded_protein_ref'].Abs, dtype=np.float32)
n = np.array(df[df['sample'] == nucleotide].Abs)
nm = np.array(df[df['sample'] == nucleotide].nm)


# function for summing multiple spectra, each scaled by a concentration
# parameter
def combine_spectra(spectra, *concentrations):
    return sum(k * s for s, k in zip(spectra, concentrations))


# fit a combined spectrum, composed of scaled protein and nucleotide
# reference spectra, to the observed spectrum
spectra = np.vstack([p, n])
conc = curve_fit(combine_spectra, spectra, pn, p0=[1, 1])[0]

# report the optimal scale factors for the best-fit combined spectrum.
# these are the concentration values of protein and nucleotide,
# respectively, in the observed spectrum
print(conc)

# extract the index position of the 280 and 260 datapoints in our vectors,
# so that the 280/260 ratio can be calculated.
nm280 = np.nonzero(nm == 280.0)[0][0]
nm260 = np.nonzero(nm == 260.0)[0][0]

# construct a vector for each separated spectrum (protein and nucleotide)
# by scaling the reference spectra up by the best-fit scale factors.
protein_corr = conc[0] * p
nuc_corr = conc[1] * n

# plot the constructed spectra alongside the observed spectrum
plt.plot(nm, pn, 'k-', label='Protein + Nuc')
plt.plot(nm, protein_corr, 'b-', label='Protein (k={})'.format(conc[0]))
plt.plot(nm, nuc_corr, 'r-', label='Nuc (k={})'.format(conc[1]))
plt.plot(nm, combine_spectra(spectra, *conc), 'k--', label='Best Fit')
plt.legend(loc='best')
plt.xlim(min(nm), max(nm))
plt.xlabel('wavelength (nm)')
plt.ylabel('10 mm Absorbance')
plt.title(output)

# save the figure to the specified output
plt.savefig(output)

# report the corrected protein concentration in uM, using the extinction coeff
print('Protein conc/uM:', 1e6*(protein_corr[nm280]/(29910*1)))
# report the 260/280 ratio of the observed spectrum
print('260/280 ratio: ', pn[nm260]/pn[nm280])
