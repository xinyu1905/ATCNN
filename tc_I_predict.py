# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
import sys

ChemicalSymbols = [ 'X',  'H',  'He', 'Li', 'Be','B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si','P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Sc', 'Ti', 'V',  'Cr','Mn', 'Fe', 'Co', 'Ni', 'Cu',
                    'Zn', 'Ga', 'Ge', 'As', 'Se','Br', 'Kr', 'Rb', 'Sr', 'Y',
                    'Zr', 'Nb', 'Mo', 'Tc', 'Ru','Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe','Cs', 'Ba', 'La', 'Ce', 'Pr',
                    'Nd', 'Pm', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W','Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po','At', 'Rn', 'Fr', 'Ra', 'Ac',
                    'Th', 'Pa', 'U',  'Np', 'Pu','Am', 'Cm', 'Bk', 'Cf', 'Es',
                    'Fm', 'Md', 'No', 'Lr']
atomicPeriod = [0, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                2, 3, 3, 3, 3, 3, 3, 3, 3, 4,
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
                5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
                7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 7, 7]
atomicGroup = [0,  1, 18,  1,  2, 13, 14, 15, 16, 17,
              18,  1,  2, 13, 14, 15, 16, 17, 18,  1,
               2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
              12, 13, 14, 15, 16, 17, 18,  1,  2,  3,
               4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
              14, 15, 16, 17, 18,  1,  2,  3,  3,  3,
               3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
               3,  3,  4,  5,  6,  7,  8,  9, 10, 11,
              12, 13, 14, 15, 16, 17, 18,  1,  2,  3,
               3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
               3,  3,  3,  3]
atomicNum = {}
for anum, symbol in enumerate(ChemicalSymbols):
    atomicNum[symbol] = anum

# atom 
class atom:
    def __init__(self,symbol,num=1):
        self.name = symbol
        self.atomicNum = atomicNum[symbol]
        self.atomicPeriod = atomicPeriod[atomicNum[symbol]]
        self.atomicGroup = atomicGroup[atomicNum[symbol]]
        self.num = num
# read the compounds
def readComponent(comps):
    namelist = []
    numlist = []
    ccomps = comps
    while(len(ccomps) != 0):
        stemp = ccomps[1:]
        if(len(stemp) == 0):
            namelist.append(ccomps)
            numlist.append(1.0)
            break
        it = 0
        for st in stemp:
            it = it + 1
            if(st.isupper()):
                im = 0
                for mt in stemp[:it]:
                    im = im + 1
                    if(mt.isdigit()):
                        namelist.append(ccomps[0:im])
                        numlist.append(float(ccomps[im:it]))
                        ccomps = ccomps[it:]
                        break
                    elif(im == len(stemp[:it])):
                        namelist.append(ccomps[0:im])
                        numlist.append(1.0)
                        ccomps = ccomps[it:]
                        break
                break
            elif(it == len(stemp)):
                im = 0
                for mt in stemp:
                    im = im + 1
                    if(mt.isdigit()):
                        namelist.append(ccomps[0:im])
                        numlist.append(float(ccomps[im:]))
                        ccomps = ccomps[it+1:]
                        break
                    elif(im == len(stemp)):
                        namelist.append(ccomps)
                        numlist.append(1.0)
                        ccomps = ccomps[it+1:]
                        break
                break
    return namelist, numlist

#creat atom vector
def get_component_vector(comps):
    namelist, numlist = readComponent(comps)
    avector = 100 * [0]
    asum = sum(numlist)
    for i in range(len(namelist)):
        at = atom(namelist[i])
        avector[at.atomicNum-1] = numlist[i]/asum
    return avector

# load the trained model for prediction
def predict_tc(mat):
    model = load_model('tc_model_I.h5')
    avmat = get_component_vector(mat)
    avmat = np.array(avmat)
    avmat = np.reshape(avmat,(1,10,10,1))
    tc = model.predict(avmat)
    return tc[0][0]

if __name__ == '__main__':
    tc = predict_tc(sys.argv[1])
    #tc = predict_tc('MgB2')
    print(tc)