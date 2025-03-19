import os

CWD = os.getcwd()
PDR_CDIR = '/home/awinter/PDR1.5.4_210817_rev2095/'
IDAT_PATH = PDR_CDIR+'AnalysisTools/IDAT/idat'
CHEM_DIR = 'data/Chimie/'
LINE_DIR = 'data/Lines/'
STMODS_DIR = os.path.dirname(os.path.realpath(__file__))+'/'


pc2cm = 3.086e18
Msol2g = 1.988e33
au2cm =  1.496e13
mH = 1.6738e-24
mu = 2.3
G = 6.67e-8
Msol  =1.989e33
Rsol = 6.957e10
au = 1.496e13
sig_SB = 5.67e-5 #K^-4
mp = 1.6726e-24
k_B = 1.81e-16 #cgs K^-1
Lsol = 3.828e33 #erg/s
year = 365.*24.*60.0*60.0

#Rsol in cm
Rsol = 6.957e10
h = 6.626e-27  # Planck's constant in erg*s
c = 2.998e10   # Speed of light in cm/s
eV = 1.602e-12