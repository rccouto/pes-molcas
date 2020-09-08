#!/usr/bin/env python3

###!/usr/bin/env nix-shell
###!nix-shell -i python -p "python37.withPackages(ps: with ps; [ numpy toolz matplotlib])"

import math, re, optparse, operator, os, glob
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import itertools


def load(file):
    f = open(file, 'r')
    r = []
    a = []
    d = []
    for l in f.readlines():
        try:
            n = list(map(str,l.replace(',',' ').split()))
            
            if len(n)>0:
                if n[0].startswith('R'):
                    r.append(n)
                elif n[0].startswith('A'):
                    a.append(n)
                elif n[0].startswith('D'):
                    d.append(n)
        except ValueError:
            pass
    return (np.array(r), np.array(a), np.array(d)) 

def readxyz(filename):
    xyzf = open(filename, 'r')
    xyzarr = np.zeros([1, 3])
    atomnames = []
    if not xyzf.closed:
        # Read the first line to get the number of particles
        npart = int(xyzf.readline())
        # and next for title card
        title = xyzf.readline()

        # Make an N x 3 matrix of coordinates
        xyzarr = np.zeros([npart, 3])
        i = 0
        for line in xyzf:
            words = line.split()
            if (len(words) > 3):
                atomnames.append(words[0])
                xyzarr[i][0] = float(words[1])
                xyzarr[i][1] = float(words[2])
                xyzarr[i][2] = float(words[3])
                i = i + 1
    return (xyzarr, atomnames)

def readzmat(filename):
    zmatf = open(filename, 'r')
    atomnames = []
    rconnect = []
    rlist = []
    aconnect = []
    alist = []
    dconnect = []
    dlist = []
    variables = {}
    
    if not zmatf.closed:
        for line in zmatf:
            words = line.split()
            eqwords = line.split('=')
            
            if len(eqwords) > 1:
                varname = str(eqwords[0]).strip()
                try:
                    varval  = float(eqwords[1])
                    variables[varname] = varval
                except:
                    print("Invalid variable definition: " + line)
            
            else:
                if len(words) > 0:
                    atomnames.append(words[0])
                if len(words) > 1:
                    rconnect.append(int(words[1]))
                if len(words) > 2:
                    rlist.append(words[2])
                if len(words) > 3:
                    aconnect.append(int(words[3]))
                if len(words) > 4:
                    alist.append(words[4])
                if len(words) > 5:
                    dconnect.append(int(words[5]))
                if len(words) > 6:
                    dlist.append(words[6])
    
    return (atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist) 


def write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist):
    npart = len(atomnames)

    xyzarr = np.zeros([npart, 3])
    if (npart > 1):
        xyzarr[1] = [rlist[0], 0.0, 0.0]

    if (npart > 2):
        i = rconnect[1] - 1
        j = aconnect[0] - 1
        r = rlist[1]
        theta = alist[0] * np.pi / 180.0
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        a_i = xyzarr[i]
        b_ij = xyzarr[j] - xyzarr[i]
        if (b_ij[0] < 0):
            x = a_i[0] - x
            y = a_i[1] - y
        else:
            x = a_i[0] + x
            y = a_i[1] + y
        xyzarr[2] = [x, y, 0.0]

    for n in range(3, npart):
        r = rlist[n-1]
        theta = alist[n-2] * np.pi / 180.0
        phi = dlist[n-3] * np.pi / 180.0
        
        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * cosTheta
        y = r * cosPhi * sinTheta
        z = r * sinPhi * sinTheta
        
        i = rconnect[n-1] - 1
        j = aconnect[n-2] - 1
        k = dconnect[n-3] - 1
        a = xyzarr[k]
        b = xyzarr[j]
        c = xyzarr[i]
        
        ab = b - a
        bc = c - b
        bc = bc / np.linalg.norm(bc)
        nv = np.cross(ab, bc)
        nv = nv / np.linalg.norm(nv)
        ncbc = np.cross(nv, bc)
        
        new_x = c[0] - bc[0] * x + ncbc[0] * y + nv[0] * z
        new_y = c[1] - bc[1] * x + ncbc[1] * y + nv[1] * z
        new_z = c[2] - bc[2] * x + ncbc[2] * y + nv[2] * z
        xyzarr[n] = [new_x, new_y, new_z]

    #for i in range(npart):
    #    print('{:<4s}\t{:>11.5f}\t{:>11.5f}\t{:>11.5f}'.format(atomnames[i], xyzarr[i][0], xyzarr[i][1], xyzarr[i][2]))
        
    return xyzarr


def write_zmat(xyzarr, atomnames):
    distmat=distance_matrix(xyzarr)
    npart, ncoord = xyzarr.shape
    rlist = []
    rconnect= []
    alist = []
    aconnect= []
    dlist = []
    dconnect = []
    
    if npart > 1:
        rlist.append(distmat[0][1])
        rconnect.append(1)
    
        if npart > 2:
            rlist.append(distmat[0][2])
            rconnect.append(1)
            alist.append(angle(xyzarr, 2, 0, 1))
            aconnect.append(2)
                
            if npart > 3:
                for i in range(3, npart):
                    rlist.append(distmat[i-3][i])
                    rconnect.append(i-2)
                    alist.append(angle(xyzarr, i, i-3, i-2))
                    aconnect.append(i-1)
                    dlist.append(dihedral(xyzarr, i, i-3, i-2, i-1))
                    dconnect.append(i)
                        
    return rlist, rconnect, alist, aconnect, dlist, dconnect

def distance_matrix(xyzarr):
    npart, ncoord = xyzarr.shape
    dist_mat = np.zeros([npart, npart])
    for i in range(npart):
        for j in range(0, i):
            rvec = xyzarr[i] - xyzarr[j]
            dist_mat[i][j] = dist_mat[j][i] = np.sqrt(np.dot(rvec, rvec))
    #print( dist_mat)
    return dist_mat

def angle(xyzarr, i, j, k):
    rij = xyzarr[i] - xyzarr[j]
    rkj = xyzarr[k] - xyzarr[j]
    cos_theta = np.dot(rij, rkj)
    sin_theta = np.linalg.norm(np.cross(rij, rkj))
    theta = np.arctan2(sin_theta, cos_theta)
    theta = 180.0 * theta / np.pi
    return theta

def dihedral(xyzarr, i, j, k, l):
    rji = xyzarr[j] - xyzarr[i]
    rkj = xyzarr[k] - xyzarr[j]
    rlk = xyzarr[l] - xyzarr[k]
    v1 = np.cross(rji, rkj)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(rlk, rkj)
    v2 = v2 / np.linalg.norm(v2)
    m1 = np.cross(v1, rkj) / np.linalg.norm(rkj)
    x = np.dot(v1, v2)
    y = np.dot(m1, v2)
    chi = np.arctan2(y, x)
    chi = -180.0 - 180.0 * chi / np.pi
    if (chi < -180.0):
        chi = chi + 360.0
    return chi
            

def set_coordinates(rinlist,ainlist,dinlist, rlist, alist, dlist):
    r = []
    a = []
    d = []


    if len(rinlist) == 0:
        r = [0]*len(rlist)
    else:
        for i in range(len(rlist)):
            key = False
            for j in range(len(rinlist)):
                if rlist[i] == rinlist[j][0]:
                    r.append(float(rinlist[j][1]))
                    key = True
                if j == (len(rinlist)-1) and key == False:
                    r.append(0)

    if len(ainlist) == 0:
        a = [0]*len(alist)
    else:
        for i in range(len(alist)):
            key = False
            for j in range(len(ainlist)):
                if alist[i] == ainlist[j][0]:
                    a.append(float(ainlist[j][1]))
                    key = True
                if j == (len(ainlist)-1) and key == False:
                    a.append(0.0)

    if len(dinlist) == 0:
        a = [0]*len(dlist)
    else:
        for i in range(len(dlist)):
            key = False
            for j in range(len(dinlist)):
                if dlist[i] == dinlist[j][0]:
                    d.append(float(dinlist[j][1]))
                    key = True
                if j == (len(dinlist)-1) and key == False:
                    d.append(0.0)

    return (r, a, d)

def generate_coord(Coord,NM,Q):
    NewCoord = []
    for j in range(len(Coord)):
        NewCoord.append(Coord[j] + (NM[j]*Q))
        
    return NewCoord



# Set the Input Name
def InputName(file, i, j):
    if j == None:
        if i > 99:
            var = "-%s.input" % i
        elif i > 9:
            var = "-0%s.input" % i
        else:
            var = "-00%s.input" % i

    else:
        if i > 99 and j > 99:
            var = "-%s.%s.input" % (i,j)
            
        elif i > 99 and j > 9:
            var = "-%s.0%s.input" % (i,j)
            
        elif i > 9 and j > 9:
            var = "-0%s.0%s.input" % (i,j)
            
        elif i > 9 and j > 99:
            var = "-0%s.%s.input" % (i,j)
            
        elif i < 10 and j > 9:
            var = "-00%s.0%s.input" % (i,j)

        elif i > 9 and j < 10:
            var = "-0%s.00%s.input" % (i,j)
        
        else:
            var = "-00%s.00%s.input" % (i,j)
    
    input=file.rsplit( ".", 1 )[ 0 ] + var
    
    return input



# Write Input File
def WriteInput(input, bs, symb, NewCoord, sym, Q, Q2):
    # Write in Input File
    inp = open(input, 'w')
    inp.write("&GATEWAY \n")
    if Q2 == None:
        inp.write("Title = Q: %.6s \n" % Q)
    else:
        inp.write("Title = Q1: %.6s  Q2: %.6s \n" % (Q, Q2))
    if sym:
        inp.write("Symmetry \n %s \n" % sym)
    inp.write("Basis set\n")

    h=1
    c=1
    o=1
    n=1
    f=1
    b=1
    for j in range(len(NewCoord)):
        # End of Basis
        if j != 0 and symb[j] != symb[j-1]:
            inp.write("End of basis\nBasis set\n")
                
        if symb[j]=="C":
            if c==1:
                inp.write("C.%s\n" % bs)
                inp.write(" %s1    %s Angstrom \n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))
                c=2
            else:
                inp.write(" %s%s    %s Angstrom \n" % (''.join(map(str,symb[j])), c, '  '.join(map(str,NewCoord[j]))))
                c=c+1
        elif symb[j]=="O":
            if o==1:
                inp.write("O.%s\n" % bs)
                inp.write(" %s1    %s Angstrom\n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))
                o=2
            else:
                inp.write(" %s%s    %s Angstrom\n" % (''.join(map(str,symb[j])), o, '  '.join(map(str,NewCoord[j]))))
                o=o+1
        elif symb[j]=="H":
            if h==1:
                inp.write("H.%s\n" % bs)
                inp.write(" %s1    %s Angstrom\n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))
                h=2
            else:
                inp.write(" %s%s    %s Angstrom\n" % (''.join(map(str,symb[j])), h, '  '.join(map(str,NewCoord[j]))))
                h=h+1
        elif symb[j]=="N":
            if n==1:
                inp.write("N.%s\n" % bs)
                inp.write(" %s1    %s Angstrom\n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))
                n=2
            else:
                inp.write(" %s%s    %s Angstrom\n" % (''.join(map(str,symb[j])), n, '  '.join(map(str,NewCoord[j]))))
                n=n+1
        elif symb[j]=="F":
            if f==1:
                inp.write("F.%s\n" % bs)
                inp.write(" %s1    %s Angstrom\n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))
                f=2
            else:
                inp.write(" %s%s    %s Angstrom\n" % (''.join(map(str,symb[j])), f, '  '.join(map(str,NewCoord[j]))))
                f=f+1
        elif symb[j]=="B":
            if b==1:
                inp.write("B.%s\n" % bs)
                inp.write(" %s1    %s Angstrom\n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))
                b=2
            else:
                inp.write(" %s%s    %s Angstrom\n" % (''.join(map(str,symb[j])), b, '  '.join(map(str,NewCoord[j]))))
                b=b+1
        elif symb[j]=="X":
            with open("rydberg.basis") as fmolcas:
                inp.write(fmolcas.read())
                inp.write(" %s    %s Angstrom\n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))
                fmolcas.close()
        else:
            print ("Unknown %s Element!!!" % symb[j])
            exit(0)
                
        # Closing End of Basis
        if j == (len(NewCoord)-1):
            inp.write("End of basis\n")

    inp.write("&SEWARD &END \nEnd of input\n")
    with open("molcas.input") as fmolcas:
        inp.write(fmolcas.read())
        fmolcas.close()
        inp.close()
            

# Write Molecular Movie
def WriteMovie(inp, symb, NewCoord,Q1, Q2):
    inp.write(" %s \n" % len(NewCoord) )
    
    if Q2 != False:
        inp.write("Q1 = %.4s  Q2 = %.4s\n" % (Q1, Q2) )
    else:
        inp.write("Q = %.4s \n" % Q1 )        
        
    for j in range(len(NewCoord)):
        inp.write(" %s    %s \n" % (''.join(map(str,symb[j])), '  '.join(map(str,NewCoord[j]))))



def get_energy(level, nroots):
    E=[]
    #c=0
    files=sorted(glob.iglob('*.log')) # Used to get all files in numerical order
    for file in files:
        energy=0.0
        for i in open( file ).readlines():
            
            if level == "RASSCF":
                if re.search(r"::    RASSCF root number", i) is not None: # Find energy in .log
                    words = i.split()
                    energy = float( words[7] )  # Energy is the sixth word
                    E.append(energy)
                    
            elif level == "CASPT2":
                if re.search(r"::    CASPT2 Root", i) is not None: # Find energy in .log
                    words = i.split()
                    energy = float( words[6] )  # Energy is the sixth word
                    E.append(energy)
                    
            elif level == "MS-CASPT2":
                if re.search(r":    MS-CASPT2 Root", i) is not None: # Find energy in .log
                    words = i.split()
                    energy = float( words[6] )  # Energy is the sixth word
                    E.append(energy)
                    
            elif level == "XMS-CASPT2":
                if re.search(r":    XMS-CASPT2 Root", i) is not None: # Find energy in .log
                    words = i.split()
                    energy = float( words[6] )  # Energy is the sixth word
                    E.append(energy)
                    
            elif level == "RASSI":
                if re.search(r"::    RASSI State ", i) is not None: # Find energy in .log
                    words = i.split()
                    energy = float( words[6] )  # Energy is the sixth word
                    E.append(energy)
            else:
                print("You forgot something, right? Maybe... level of calculation???")
                exit()
            
        if energy == 0.0:
            for j in range(nroots):
                E.append(energy)

    return E


def get_2D_Q():
    Q1=[]
    Q2=[]
    files=sorted(glob.iglob('*.log')) 
    for file in files:
        for i in open( file ).readlines():
            if re.search(r"Title = ", i) is not None:
                words = i.split()
                q1 = str( words[3] ) 
                q2 = str( words[5] )
                Q1.append(q1)
                Q2.append(q2)
    return Q1, Q2

def get_1D_Q():
    Q=[]
    files=sorted(glob.iglob('*.log')) 
    for file in files:
        for i in open( file ).readlines():
            if re.search(r"Title = ", i) is not None:
                words = i.split()
                q = str( words[3] ) 
                Q.append(q)
    return Q
                    
def project_to_nm(eq, ref, nm, mass):
    
    ceq  = center_structure(eq, mass)
    cref = center_structure(ref, mass)

    
    
def transform_mol(xyz, atomnames, mass):
    
    com = xyz / mass
    print(com)
    center = xyz - com
    print(center)
    rlist, rconnect, alist, aconnect, dlist, dconnect= write_zmat(center, atomnames)
    txyz = write_xyz(atomnames, rconnect, rlist, aconnect, alist, dconnect, dlist)

    return txyz

def print_coord(atomnames, xyzarr):
    print(len(atomnames))
    print("Coordinates")
    for i in range(len(atomnames)):
        print('{:<4s}\t{:>11.5f}\t{:>11.5f}\t{:>11.5f}'.format(atomnames[i], xyzarr[i][0], xyzarr[i][1], xyzarr[i][2]))


def get_nac(file,natoms):
    SYMB=[]
    NAC=[]
    GEO=[]
    CSF=[]
    energy=[]
    
    data = re.compile(r'\s(\S+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)')
    flag=False
    RASSCF=False
    flagCSF=False
    c1=0
    c2=0
    c3=0
    with open(file, "r") as log:
        for line in log:
            
            if line.startswith("      Header of the ONEINT file:"):
                RASSCF=True
            if RASSCF:
                if data.search(line):
                    symb = str(data.search(line).group(1) )
                    symb = re.sub(r"[^A-Za-z]+", '', symb)
                    SYMB.append(symb)
                    x = float(data.search(line).group(2) ) # Get the X Coordinate
                    GEO.append(x)
                    y = float(data.search(line).group(3) ) # Get the Y Coordinate
                    GEO.append(y)
                    z = float(data.search(line).group(4) ) # Get the Z Coordinate
                    GEO.append(z)
                    c1=c1+1
                    if c1 == natoms:
                        RASSCF=False

            if line.startswith(" *              CSF derivative coupling               *"):
                flagCSF=True

            if flagCSF:
                if data.search(line):
                    x = float(data.search(line).group(2) ) # Get the X Coordinate
                    CSF.append(x)
                    y = float(data.search(line).group(3) ) # Get the Y Coordinate
                    CSF.append(y)
                    z = float(data.search(line).group(4) ) # Get the Z Coordinate
                    CSF.append(z)
                    c3=c3+1
                if c3 == natoms:
                     flagCSF=False

            if re.search(r"Energy difference:", line) is not None: # Find energy in .log
                words = line.split()
                energy = float( words[2] )  # Energy is the sixth word

                
            if line.startswith(" *              Total derivative coupling              *"):
                flag=True

            if flag:
                if data.search(line):
                    x = float(data.search(line).group(2) ) # Get the X Coordinate
                    NAC.append(x)
                    y = float(data.search(line).group(3) ) # Get the Y Coordinate
                    NAC.append(y)
                    z = float(data.search(line).group(4) ) # Get the Z Coordinate
                    NAC.append(z)
                    c2=c2+1
                if c2 == natoms:
                    break
    
    log.close()
    GEO=np.array(GEO).reshape(-1,3)
    NAC=np.array(NAC).reshape(-1,3)
    CSF=np.array(CSF).reshape(-1,3)    
    return SYMB, GEO, NAC, CSF, energy



    



# MAIN PROGRAM
def main():
    import sys
    f = optparse.OptionParser(usage="usage: %prog [options] filename")
    # Get Type of Run
    f.add_option('--zmatrix', action="store_true", default=False, help='Generate PES using Z-matrix coordinates')
    f.add_option('--xyz', action="store_true", default=False, help='Generate PES using cartesian XYZ coordinates')
    f.add_option('--proj', action="store_true", default=False, help='Projection onto normal modes')
    f.add_option('--get', action="store_true", default=False, help='Get the PES from OpenMolcas')
    f.add_option('--get_nm', action="store_true", default=False, help='Stuff')
    f.add_option('--getnac', action="store_true", default=False, help='Get the Non-adiabatic coupling vectors from OpenMolcas')
    f.add_option('--getnac1', action="store_true", default=False, help='Non-adiabatic coupling vectors from SINGLE OpenMolcas log file')
    f.add_option('--getdqv', action="store_true", default=False, help='Get Diabatic potentials and coupling from DQV method on OpenMolcas')
    # Get Z-matrix Coordinates
    f.add_option('-z', '--zmat', type = str, default = None, help='Give the Zmat structure')
    # Get Normal Coordinates
    f.add_option('-n', '--nm' , type = str, default = 'co2-nm.coord', help='Normal mode coordinates')
    # Get Normal Coordinates
    f.add_option('--nm2' , type = str, default = None, help='Second Normal mode coordinate')    
    # Get Equilibrium geometry Coordinates
    f.add_option('-e', '--eq' , type = str, default = None, help='Equilibrium geometry')
    # Get Equilibrium geometry Coordinates
    f.add_option('-g', '--grad' , type = str, default = None, help='Gradient')
    # Reference structure
    f.add_option('-r', '--ref' , type = str, default = None, help='Reference structure for nm-proj')   
    # Get Number of Steps
    f.add_option('-N', '--N'  , type = int, default = 30, help='Number of steps in PES')
    # Get Number of Steps
    f.add_option('--N2'  , type = int, default = 30, help='Number of steps in PES')    
    # Get Initial Amplitude
    f.add_option('-i', '--qi'  , type = float, default = -3, help='Initial Q')
    # Get Final Amplitude
    f.add_option('-f', '--qf'  , type = float, default = 3, help='Final Q')
    # Get Initial Amplitude
    f.add_option('--qi2'  , type = float, default = -3, help='Initial Q')
    # Get Final Amplitude
    f.add_option('--qf2'  , type = float, default = 3, help='Final Q') 
    # Get Number of States
    f.add_option('-s', '--st' , type = int, default = 1, help='Number of states used in get mode')    
    # Basis set
    f.add_option('-b', '--bs' , type = str, default = 'ANO-RCC-VTZP', help='Basis set for OpenMolcas')
    # Symmetry
    f.add_option('-c', '--sym' , type = str, default = None, help='Symmetry constrain in OpenMolcas')
    # Just movie
    f.add_option('--movie', action="store_true", default=False, help='Print just the NM movie')
    # Level of calculation
    f.add_option( '-l', '--level' , type = str, default = None, help='Level of calculation to get results, i.e., RASSCF, CASPT2...')
    f.add_option('--diff', action="store_true", default=False, help='Compute the difference between two structures (use --proj )')
    f.add_option('--sum', action="store_true", default=False, help='Add the NM projection to the equilibrium geomtry (use --proj )')
    f.add_option('--rotate', action="store_true", default=False, help='Generate PES using cartesian XYZ coordinates')
    f.add_option('--td', action="store_true", default=False, help='2D PES calculation')
    f.add_option('--natoms', type = int, default = None, help='Number of atoms in molecule')
    f.add_option('--log' , type = str, default = None, help='OpenMolcas log file')
    f.add_option('--atm1' , type = str, default = None, help='Atom 1 for 1D potential ')
    (arg, args) = f.parse_args(sys.argv[1:])

    if len(sys.argv) == 1:
        f.print_help()
        sys.exit(1)


    if arg.zmatrix == True:

        atomnames,rconnect,rlist,aconnect,alist,dconnect,dlist=readzmat(arg.zmat)
        reqlist,aeqlist,deqlist=load(arg.eq)
        rnmlist, anmlist, dnmlist=load(arg.nm)
        
        req,aeq,deq=set_coordinates(reqlist,aeqlist,deqlist, rlist, alist, dlist)
        rnm,anm,dnm=set_coordinates(rnmlist,anmlist,dnmlist, rlist, alist, dlist)

        # Get Step Ratio
        if (arg.N % 2 == 0):    # The N. Steps must be even, to have the Equilibrium
            arg.N = arg.N + 1   # Geometry in the center of the Curve

        Q=np.linspace(arg.qi,arg.qf,arg.N)

        if arg.td == True:
            rnm2list, anm2list, dnm2list = load(arg.nm2)
            rnm2,anm2,dnm2 = set_coordinates(rnm2list,anm2list,dnm2list, rlist, alist, dlist)
            Q2 = np.linspace(arg.qi2,arg.qf2,arg.N)
            
        # Write XYZ Files
        file=arg.nm
        inp = open("movie.dat", 'w')



        for i in range(arg.N):
            
            NewR=generate_coord(req, rnm, Q[i])
            NewA=generate_coord(aeq, anm, Q[i])
            NewD=generate_coord(deq, dnm, Q[i])           

            if arg.td == True:
                for j in range(arg.N):

                    NewR2d=generate_coord(NewR, rnm2, Q2[j])
                    NewA2d=generate_coord(NewA, anm2, Q2[j])
                    NewD2d=generate_coord(NewD, dnm2, Q2[j]) 
                    xyz2d=write_xyz(atomnames, rconnect, NewR2d, aconnect, NewA2d, dconnect, NewD2d)
                    WriteMovie(inp, atomnames, xyz2d, Q[i], Q2[j])
                
            else:
                xyz=write_xyz(atomnames, rconnect, NewR, aconnect, NewA, dconnect, NewD)
                WriteMovie(inp, atomnames, xyz, Q[i], False)
            


            if arg.movie == False:
                input=InputName(file,i,0)
                
                WriteInput(input, arg.bs, atomnames, xyz, arg.sym, Q[i])
                
                job=input.replace(".input", ".sh")

                scpt=open(job, 'w')

                with open("submit.sh") as fugu:
                    scpt.write(fugu.read())
                scpt.write("pymolcas -b 1 -f %s" % input)
                fugu.close()
                scpt.close()

                #call("sbatch %s" % job, shell=True)




    elif arg.xyz == True:        
        eqxyz, atomnames=readxyz(arg.eq)
        nmxyz,dump=readxyz(arg.nm)

        
        # Get Step Ratio
        if (arg.N % 2 == 0):    # The N. Steps must be even, to have the Equilibrium
            arg.N = arg.N + 1   # Geometry in the center of the Curve

                    
        if arg.td == True:
            nm2xyz,dump=readxyz(arg.nm2)
            Q2=np.linspace(arg.qi2,arg.qf2,arg.N)
        # Write XYZ Files
        file=arg.nm
        inp = open("movie.dat", 'w')
        
        Q=np.linspace(arg.qi,arg.qf,arg.N)

        for i in range(arg.N):
            NewXYZ=generate_coord(eqxyz, nmxyz, Q[i])
            
            if arg.td == True:
                for j in range(arg.N):
                    TdXYZ=generate_coord(NewXYZ, nm2xyz, Q2[j])
                    WriteMovie(inp, atomnames, TdXYZ, Q[i], Q2[j])
                    if arg.movie == False:
                        input=InputName(file,i,j)
                        WriteInput(input, arg.bs, atomnames, TdXYZ, arg.sym, Q[i], Q2[j])
                
                        job=input.replace(".input", ".sh")
                        scpt=open(job, 'w')
                
                        with open("submit.sh") as fugu:
                            scpt.write(fugu.read())
                        scpt.write("pymolcas -b 1 -f %s" % input)
                        fugu.close()
                        scpt.close()

                    #call("sbatch %s" % job, shell=True)
        
                    
            else:
                WriteMovie(inp, atomnames, NewXYZ, Q[i], False)

                if arg.movie == False:
                    input=InputName(file,i,None)
                    WriteInput(input, arg.bs, atomnames, NewXYZ, arg.sym, Q[i], None)
                
                    job=input.replace(".input", ".sh")
                    scpt=open(job, 'w')
                
                    with open("submit.sh") as fugu:
                        scpt.write(fugu.read())
                    scpt.write("pymolcas -b 1 -f %s" % input)
                    fugu.close()
                    scpt.close()
                    
                    #call("sbatch %s" % job, shell=True)
        



            
    elif arg.proj == True:

        
        eqxyz,atomnames=readxyz(arg.eq)
        #nmxyz,test=readxyz(arg.nm)
        refxyz,dump=readxyz(arg.ref)

        
        diff=eqxyz-refxyz

        #if arg.diff == True:
        print_coord(atomnames,diff)
        
        #c= nmxyz/diff
        #print_coord(atomnames,c)

        #if arg.sum == True:
        #    sum=eqxyz+c
        #    print_coord(atomnames,sum)
        

    elif arg.rotate == True:

        
        eqxyz,atomnames=readxyz(arg.eq)
        #nmxyz,test=readxyz(arg.nm)
        refxyz,dump=readxyz(arg.ref)
        gradxyz,dump=readxyz(arg.grad)

        add=refxyz+gradxyz
        print_coord(atomnames,add)
        
        #call("calculate_rmsd.py %s %s --print" % (arg.eq, arg.ref), shell=True)
        #call("calculate_rmsd.py %s %s --print > %s" % (arg.eq, arg.ref, "ref1.dat"), shell=True)
        
        
    elif arg.get_nm == True:

        rm=109
        atomnames,rconnect,rlist,aconnect,alist,dconnect,dlist=readzmat(arg.zmat)
        reqlist,aeqlist,deqlist=load(arg.eq)
        rnmlist, anmlist, dnmlist=load(arg.nm)
        
        req,aeq,deq=set_coordinates(reqlist,aeqlist,deqlist, rlist, alist, dlist)
        rnm,anm,dnm=set_coordinates(rnmlist,anmlist,dnmlist, rlist, alist, dlist)

        equilibrium=write_xyz(atomnames, rconnect, req, aconnect, aeq, dconnect, deq)
        
        NewR=generate_coord(req, rnm, 1)
        NewA=generate_coord(aeq, anm, 1)
        NewD=generate_coord(deq, dnm, 1)           
            
        added_nm=write_xyz(atomnames, rconnect, NewR, aconnect, NewA, dconnect, NewD)

        normal_mode=(equilibrium-added_nm)

        print_coord(atomnames,normal_mode)
        
        
    # GET PES
    elif arg.get == True:

        nroots=arg.st

        # Get energy from .log Files
        energy=get_energy(arg.level, nroots)

                
        # Get Step Ratio
        if (arg.N % 2 == 0):    # The N. Steps must be even, to have the Equilibrium
            arg.N = arg.N + 1   # Geometry in the center of the Curve

        
        if arg.td == True:

            Q1=np.linspace(arg.qi,arg.qf,arg.N)
            Q2=np.linspace(arg.qi2,arg.qf2,arg.N)
            q1=[]
            q2=[]
            for i in range(arg.N):
                for j in range(arg.N):
                    q1.append(Q1[i])
                    q2.append(Q2[j])
        else:
            #q1=np.linspace(arg.qi,arg.qf,arg.N)
            q1=get_1D_Q()

        x1=[]
        x2=[]
        y=[]
        for i in range(len(q1)):
            for j in range(nroots):
                if energy[(nroots*i)+j] == 0.0:
                    pass
                else:
                    y.append(energy[(nroots*i)+j])
                    if j == 0:
                        x1.append(q1[i])
                        if arg.td == True:
                            x2.append(q2[i])

                            
        for i in range(len(x1)):
            print ("%.6s" % x1[i], end=' ')
            if arg.td == True:
                print ("%.6s" % x2[i], end='  ')
            for j in range(nroots):
                print("%.14s" % y[(nroots*i)+j], end=' ')
            print(' ')

            if arg.td == True and x1[i] != x1[i+1] and i != (len(x1)-1):
                print(' ')


    # GET NACs
    elif arg.getnac == True:

        n=arg.N
        qxyz,dump=readxyz(arg.nm)
        #eqxyz, atomnames=readxyz(arg.eq)
        
        Q1=np.linspace(arg.qi,arg.qf,arg.N)
        Q2=np.linspace(arg.qi2,arg.qf2,arg.N)

        files=sorted(glob.iglob('*.log'))
        i=0
        j=0
        J=0
        dgdq=2.41719
        for file in files:
            symb,geoxyz,nacxyz,fcsf,energy=get_nac(file, arg.natoms)


            dq=0.00001
            Gdq=generate_coord(geoxyz, qxyz, dq)
            dq2=-0.00001
            Gdq2=generate_coord(geoxyz, qxyz, dq2)
            
            #print(np.array(geoxyz))
            #print(np.array(Gdq))

            #dgdq=(Gdq-geoxyz)/dq
            #dgdq2=(np.array(Gdq)-np.array(Gdq2))/(2*dq)
            #print(nacxyz)
            #nacme=np.sum(nacxyz)
            #nacme2=np.sum(dgdq2*nacxyz)
            #dd=dgdq*nacxyz
            #print(nacxyz)
            #nacme=np.sum(nacxyz)
            #h=(nacxyz-fcsf)*energy
            #nacme=np.sum(dgdq*h)
            
            #print(file, nacxyz)
            #print(fcsf)

            #print (file, end=' ')


            #print(np.sum(nacxyz))
            
            print ("%.6s" % Q1[i], end=' ')
            print ("%.6s" % Q2[j], end='  ')
            #print(nacme, end=' ')
            print(abs(np.sum(nacxyz)), end=' ')
            print(' ')

            
            
            j=j+1
            if j == arg.N:
                j=0
                i=i+1
                print(' ')

    elif arg.getnac1 == True:
        
        
        dgdq=2.41719
        symb,geoxyz,nacxyz,fcsf,energy=get_nac(arg.log, arg.natoms)
        print(np.sum(nacxyz*dgdq))
        

    elif arg.getdqv  == True:
        
        if arg.log == True:
            print("YES")

        else:
            print("Not implemented yet")
            sys.exit(1)
           


if __name__=="__main__":
    main()
