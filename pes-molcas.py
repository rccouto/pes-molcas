#!/usr/bin/env python3

###!/usr/bin/env nix-shell
###!nix-shell -i python -p "python37.withPackages(ps: with ps; [ numpy toolz matplotlib])"

import copy
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

def readpes(file):
    f = open(file, 'r')
    a = []
    for l in f.readlines():
        try:
            n = list(map(float,l.replace(',',' ').split()))
            if len(n)>0:
                a.append(n)
        except ValueError:
            pass
    f.close()
    return np.array(a)


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
        if i > 9:
            var = "-%s.input" % i
        else:
            var = "-0%s.input" % i

    else:
        if i > 9 and j > 9:
            var = "-%s.%s.input" % (i,j)
            
        elif i > 9 and j < 10:
            var = "-%s.0%s.input" % (i,j)
            
        elif i < 10 and j > 9:
            var = "-0%s.%s.input" % (i,j)
        
        else:
            var = "-0%s.0%s.input" % (i,j)
    
    input=file.rsplit( ".", 1 )[ 0 ] + var
    
    return input


# Set the Input Name
def OldInputName(file, i, j):
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

    inp.write("EPOT=0\n&SEWARD &END \nEnd of input\n")
    with open("molcas.input") as fmolcas:
        inp.write(fmolcas.read())
        fmolcas.close()
        inp.close()
            
# Write Input File
def WriteXYZ(xyz, symb, NewCoord, Q, Q2):
    
    natoms=len(NewCoord)
    inp = open(xyz, 'w')
    C=np.array(NewCoord)
    S=np.array(symb)
    
    inp.write("%s       \n" % natoms)
    if Q2 == None:
        inp.write("Q: %.6s \n" % Q)
    else:
        inp.write("Q1: %.6s  Q2: %.6s \n" % (Q, Q2))

    for i in range(len(NewCoord)):
        #inp.write(" %s  %12.10s  %12.10s  %12.10s\n" % ( S[i], C[i][0],  C[i][1],  C[i][2] ))
        inp.write('{:<4s}\t{:>14.10f}\t{:>14.10f}\t{:>14.10f} \n'.format(S[i], C[i][0], C[i][1], C[i][2]))
    inp.close()

        
# Write Molecular Movie
def WriteMovie(inp, symb, NewCoord,Q1, Q2):
    inp.write(" %s \n" % len(NewCoord) )
    
    if Q2 != False:
        inp.write("Q1 = %.4s  Q2 = %.4s\n" % (Q1, Q2) )
    else:
        inp.write("Q = %.4s \n" % Q1 )        
        
    for i in range(len(NewCoord)):
        #inp.write(" %.16s    %.45s \n" % (''.join(map(str,symb[i])), '  '.join(map(str,NewCoord[i]))))
        #inp.write(" %.16s    %.16s   %.16s   %.16s \n" % (''.join(map(str,symb[i])), NewCoord[i][0], NewCoord[i][1], NewCoord[i][2]  ) )
        inp.write('{:<4s}\t{:>14.10f}\t{:>14.10f}\t{:>14.10f} \n'.format(symb[i], NewCoord[i][0], NewCoord[i][1], NewCoord[i][2]))

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


def get_2D_Q(log):
    Q1=[]
    Q2=[]
    for i in open( log ).readlines():
        if re.search(r"Title = ", i) is not None:
            words = i.split()
            q1 = str( words[3] ) 
            q2 = str( words[5] )
            Q1.append(q1)
            Q2.append(q2)
    return Q1, Q2

def get_Q_2D():
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
                break
    return Q


def get_Q_1D(file):
    Q=[]
    for i in open( file ).readlines():
        if re.search(r"Title = ", i) is not None:
            words = i.split()
            q = str( words[3] ) 
            Q.append(q)
            break
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

def print_coord(atomnames, xyzarr, title):
    print(len(atomnames))
    print(title)
    for i in range(len(atomnames)):
        print('{:<4s}\t{:>14.10f}\t{:>14.10f}\t{:>14.10f}'.format(atomnames[i], xyzarr[i][0], xyzarr[i][1], xyzarr[i][2]))




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
                    flag=False
                    
            if re.search(r"norm:", line) is not None: # Find energy in .log
                words = line.split()
                norm = float( words[1] )  # Energy is the sixth word
    
    log.close()
    GEO=np.array(GEO).reshape(-1,3)
    NAC=np.array(NAC).reshape(-1,3)
    CSF=np.array(CSF).reshape(-1,3)    
    return SYMB, GEO, NAC, CSF, energy, norm   


def get_nac1D(file,natoms):
    SUMNAC=[]
    NAC=[]
    R=[]
    CSF=[]
    hAB=[]
    
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
                    z = float(data.search(line).group(4) ) # Get the Z Coordinate
                    c1=c1+1
                if c1 == natoms:
                    R.append(z)
                    RASSCF=False
                    c1=0

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
                    c3=0

            if re.search(r"Energy difference:", line) is not None: # Find energy in .log
                words = line.split()
                e = float( words[2] )  # Energy is the sixth word
                #energy.append(e)
                    
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
                    flag=False
                    c2=0
                    SUMNAC.append(np.sum(NAC))

                    NAC=np.array(NAC).reshape(-1,3)
                    CSF=np.array(CSF).reshape(-1,3)
                    h=(NAC-CSF)*e
                    hAB.append(np.sum(h))
                    
                    NAC=[]
                    CSF=[]
                    
    
    log.close()
    #NAC=np.array(NAC).reshape(-1,3)
    return R, SUMNAC, hAB


def get_dqv(file,natoms,nD):
    R=[]
    E=[]
    DC=[]
    Q1=[]
    Q2=[]
    W=[]
    
    data = re.compile(r'\s(\S+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)')
    dq = re.compile(r'\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)')

    RASSCF=False
    coef=False
    hamil=False
    weight=False
    
    c1=0
    c2=0
    c3=0
    c4=0
    with open(file, "r") as log:
        for line in log:

            if nD == "diatomic":
                if line.startswith("      Header of the ONEINT file:"):
                    RASSCF=True
                if RASSCF:
                    if data.search(line):
                        z = float(data.search(line).group(4) ) # Get the Z Coordinate
                        c1=c1+1
                    if c1 == natoms:
                        R.append(z)
                        RASSCF=False
                        c1=0

            elif nD == 1:
                if re.search(r"Title = ", line) is not None:
                    words = line.split()
                    q1 = float( words[3] ) 
                    Q1.append(q1)
                    
            elif nD == 2:
                if re.search(r"Title = ", line) is not None:
                    words = line.split()
                    q1 = float( words[3] ) 
                    q2 = float( words[5] )
                    Q1.append(q1)
                    Q2.append(q2)

            if line.startswith("  Diabatic Coefficients "):
                coef=True
            if coef:
                if dq.search(line):
                    x = float(dq.search(line).group(1) )
                    DC.append(x)
                    y = float(dq.search(line).group(2) )
                    DC.append(y)
                    c2=c2+1
                if c2 == natoms:
                     coef=False
                     c2=0

            if line.startswith("  Weights of adiabatic states "):
                weight=True
            if weight:
                if dq.search(line):
                    x = float(dq.search(line).group(1) ) 
                    W.append(x)
                    y = float(dq.search(line).group(2) )
                    W.append(y)
                    c4=c4+1
                if c4 == natoms:
                     weight=False
                     c4=0
                     
            if line.startswith("  Diabatic Hamiltonian  "):
                hamil=True

            if hamil:
                if dq.search(line):
                    x = float(dq.search(line).group(1) ) 
                    E.append(x)
                    y = float(dq.search(line).group(2) )
                    E.append(y)
                    
                    c3=c3+1
                if c3 == natoms:
                     hamil=False
                     c3=0
    
    log.close()
    DC=np.array(DC).reshape(-1,2)
    W=np.array(W).reshape(-1,4)
    E=np.array(E).reshape(-1,4)
    Q1=np.array(Q1)
    Q2=np.array(Q2)
    
    return R, Q1, Q2, E, DC, W





def get_nacme_molpro(file,natoms):
    SUMNAC=[]
    NAC=[]
    R=[]
    E1=[]
    E2=[]
    
    data = re.compile(r'\s(\d+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)')
    flag=False
    c1=0
    c2=0
    with open(file, "r") as log:
        for line in log:


            if re.search(r" SETTING RLIF           =", line) is not None: # Find energy in .log
                words = line.split()
                r = float( words[3] )  # Energy is the sixth word
                R.append(r)            

            if re.search(r" !MCSCF STATE  1.1 Energy", line) is not None: # Find energy in .log
                words = line.split()
                e = float( words[4] )  # Energy is the sixth word
                E1.append(e)

            if re.search(r" !MCSCF STATE  2.1 Energy", line) is not None: # Find energy in .log
                words = line.split()
                e = float( words[4] )  # Energy is the sixth word
                E2.append(e)
                
            if line.startswith(" SA-MC NACME FOR STATES 1.1 - 2.1"):
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
                    flag=False
                    c2=0
                    SUMNAC.append(np.sum(NAC))
                    NAC=[]
                    
    
    log.close()
    #NAC=np.array(NAC).reshape(-1,3)
    return R, SUMNAC, E1, E2

def get_dipole(file):
    
    data = re.compile(r'(\d+)\s+(\d+)\s+([+-]?\d+\.\d*(?:[Ee]-?\d+)?)\s+([+-]?\d+\.\d*(?:[Ee]-?\d+)?)')
    #data = re.compile(r'(\d+)\s+(\d+)\s+(-?\d+\.\d*(?:[Ee]-?\d+)?)\s+(-?\d+\.\d*(?:[Ee]-?\d+)?)\s+(-?\d+\.\d*(?:[Ee]-?\d+)?)\s+(-?\d+\.\d*(?:[Ee]-?\d+)?)')
    
    flagL=False
    flagV=False
    flagD=False
    OL = 0.0
    DM = 0.0
    OV = 0.0
    DX = 0.0
    DY = 0.0
    DZ = 0.0
    E=[]
    
    with open(file, "r") as log:
        for line in log:

            
            if re.search(r"::    RASSI State ", line) is not None: # Find energy in .log
                words = line.split()
                energy = float( words[6] )  # Energy is the sixth word
                E.append(energy)

            
            if line.startswith("++ Dipole transition strengths"):
                flagL=True
            if flagL:
                if line.startswith("         1    2") or line.startswith("         2    1"):
                    words = line.split()
                    OL = float( words[2] ) 
                    
                if line.startswith("++ Dipole transition vectors"):
                    flagL=False

            
            if line.startswith("++ Dipole transition vectors (spin-free states):"):
                flagD=True
            if flagD:
                if line.startswith("         1    2") or line.startswith("         2    1"):
                    words = line.split()
                    DX = float( words[2] ) 
                    DY = float( words[3] )
                    DZ = float( words[4] ) 
                    DM = float( words[5] )

                if line.startswith("++ Velocity transition strengths (spin-free states):"):
                    flagD=False

            
            if line.startswith("++ Velocity transition strengths (spin-free states):"):
                flagV=True
            if flagV:
                if line.startswith("         1    2") or line.startswith("         2    1"):
                    words = line.split()
                    OV = float( words[2] )  
                    break
    log.close()
    #if not O:
    #    O.append(0.0)
    #states=np.array(O).reshape(-1,3)
    
    return OL, OV, DM, DX, DY, DZ, E



def get_geo(file):
    GEO=[]
    Q1=[]
    Q2=[]
    natoms=33
    
    data = re.compile(r'\s(\S+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)\s+(-?\d+?\.\d+)')
    RASSCF=False 
    c1=0

    with open(file, "r") as log:
        for line in log:

            if line.startswith("      Header of the ONEINT file:"):
                RASSCF=True
            if RASSCF:
                if data.search(line):
                    x = float(data.search(line).group(2) ) # Get the X Coordinate
                    GEO.append(x)
                    y = float(data.search(line).group(3) ) # Get the Y Coordinate
                    GEO.append(y)
                    z = float(data.search(line).group(4) ) # Get the Z Coordinate
                    GEO.append(z)
                    c1=c1+1
                    if c1 == natoms:
                        RASSCF=False

            if re.search(r"Title = ", line) is not None:
                words = line.split()
                q1 = float( words[3] ) 
                q2 = float( words[5] )
                Q1.append(q1)
                Q2.append(q2)
                
    GEO=np.array(GEO).reshape(-1,3)
    return GEO, Q1, Q2


def center_of_mass(symb, xyz):

    #Hm = 1.007825037 #(amu)
    Hm = (1.673523708E-27)/(9.10939E-31)
    #Cm = 12.0107 #(amu)
    Cm =(1.994412767E-26)/(9.10939E-31)
    #Bm = 10.810 #(amu)
    Bm = (1.79503293E-26)/(9.10939E-31)
    #Nm = 14.00324100 #(amu)
    Nm = (2.325280177E-26)/(9.10939E-31)
    #Fm = 18.99840325 #(amu)
    Fm = (3.154741854E-26)/(9.10939E-31)
    

    COM = []
    M = []
    
    for i in range(len(symb)):
        if symb[i] == 'H':
            com = xyz[i] * Hm
            M.append(Hm)
        elif symb[i] == 'B':
            com = xyz[i] * Bm
            M.append(Bm)
        elif symb[i] == 'C':
            com = xyz[i] * Cm
            M.append(Cm)
        elif symb[i] == 'N':
            com = xyz[i] * Nm
            M.append(Nm)
        elif symb[i] == 'F':
            com = xyz[i] * Fm
            M.append(Fm)
        else:
            print("Atomic mass of %s missing" % symb[i])
            sys.exit(1)
        COM.append(com)

    MW = np.array(COM).reshape(-1,3)
    COM=COM/np.sum(M)
    M = np.array(M).reshape(-1,1)
    
    return COM, MW, M


def reduced_mass(symb, xyz):

    #Hm = 1.007825037 (amu)
    Hm = (1.673523708E-27)/(9.10939E-31)
    #Cm = 12.0107 (amu)
    Cm =(1.994412767E-26)/(9.10939E-31)
    #Bm = 10.810 (amu)
    Bm = (1.79503293E-26)/(9.10939E-31)
    #Nm = 14.00324100 (amu)
    Nm = (2.325280177E-26)/(9.10939E-31)
    #Fm = 18.99840325 (amu)
    Fm = (3.154741854E-26)/(9.10939E-31)

    RM = []

    xyz=xyz*1.88973
    
    for i in range(len(symb)):
        if symb[i] == 'H':
            t=((xyz[i][0]**2)/Hm) + ((xyz[i][1]**2)/Hm)  + ((xyz[i][2]**2)/Hm) 
        elif symb[i] == 'B':
            t=((xyz[i][0]**2)/Bm) + ((xyz[i][1]**2)/Bm)  + ((xyz[i][2]**2)/Bm) 
        elif symb[i] == 'C':
            t=((xyz[i][0]**2)/Cm) + ((xyz[i][1]**2)/Cm)  + ((xyz[i][2]**2)/Cm) 
        elif symb[i] == 'N':
            t=((xyz[i][0]**2)/Nm) + ((xyz[i][1]**2)/Nm)  + ((xyz[i][2]**2)/Nm) 
        elif symb[i] == 'F':
            t=((xyz[i][0]**2)/Fm) + ((xyz[i][1]**2)/Fm)  + ((xyz[i][2]**2)/Fm) 
        else:
            print("Atomic mass of %s missing" % symb[i])
            sys.exit(1)
        RM.append(t)
    MASS=np.sum(RM)
    
    return MASS


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def centroid(X):
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : float
        centroid
    """
    C = X.mean(axis=0)
    return C

def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """
    diff = np.array(V) - np.array(W)
    N = len(V)
    return np.sqrt((diff * diff).sum() / N)


def kabsch_rmsd(P, Q, W=None, translate=False):
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.
    An optional vector of weights W may be provided.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    W : array or None
        (N) vector, where N is points.
    translate : bool
        Use centroids to translate vector P and Q unto each other.

    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """

    if translate:
        Q = Q - centroid(Q)
        P = P - centroid(P)

    if W is not None:
        return kabsch_weighted_rmsd(P, Q, W)

    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)

def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated

    """
    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P

def align_struct(p_all, q_all):
    """
    Calculate Root-mean-square deviation (RMSD) between structure A and B, in XYZ, 
    using transformation and rotation.

    For more information, usage, example and citation read more at
    https://github.com/charnley/rmsd
    """
    p_coord = copy.deepcopy(p_all)
    q_coord = copy.deepcopy(q_all)
    
    p_cent = centroid(p_coord)
    q_cent = centroid(q_coord)
    p_coord -= p_cent
    q_coord -= q_cent
    
    # Get rotation matrix
    U = kabsch(q_coord, p_coord)
    q_all -= q_cent
    q_all = np.dot(q_all, U)
    
    # center q on p's original coordinates
    q_all += p_cent

    result_rmsd = kabsch_rmsd(p_coord, q_coord)
    
    return q_all, result_rmsd


# MAIN PROGRAM
def main():
    import sys
    f = optparse.OptionParser(usage="usage: %prog [options] filename")
    # Main modules
    f.add_option('--zmatrix', action="store_true", default=False, help='Generate PES using Z-atrix coordinates')
    f.add_option('--xyz', action="store_true", default=False, help='Generate PES using cartesian XYZ coordinates')
    f.add_option('--diff', action="store_true", default=False, help='Compute the difference between two Cartesian coordinates')
    f.add_option('--get', action="store_true", default=False, help='Get the PES from OpenMolcas')
    f.add_option('--get_nm', action="store_true", default=False, help='Stuff')
    f.add_option('--getnac', action="store_true", default=False, help='Get the Non-adiabatic coupling vectors from OpenMolcas')
    f.add_option('--getnac1d', action="store_true", default=False, help='Non-adiabatic coupling vectors from SINGLE OpenMolcas log file')
    f.add_option('--getdqv', action="store_true", default=False, help='Get Diabatic potentials and coupling from DQV method on OpenMolcas')
    f.add_option('--getnacme', action="store_true", default=False, help='Get Non-Adiabatic Coupling Matrix from Molpro')
    f.add_option('--getdm', action="store_true", default=False, help='Get oscillator strength and transition dipole moment.')
    f.add_option('--gmatrix', action="store_true", default=False, help='Create the G-matrix')
    f.add_option('--shiftpes', action="store_true", default=False, help='Shift the PES surfaces to the minium of GS')
    f.add_option('--plotpes', action="store_true", default=False, help='Fix Matlab PES')

    ### Variables
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
    f.add_option('-c', '--sym' , type = str, default = 'C1', help='Symmetry constrain in OpenMolcas')
    # Just movie
    f.add_option('--movie', action="store_true", default=False, help='Print just the NM movie')
    # Level of calculation
    f.add_option( '-l', '--level' , type = str, default = None, help='Level of calculation to get results, i.e., RASSCF, CASPT2...')
    f.add_option('--sum', action="store_true", default=False, help='Add the NM projection to the equilibrium omtry (use --proj )')
    f.add_option('--rotate', action="store_true", default=False, help='Generate PES using cartesian XYZ coordinates')
    f.add_option('--td', action="store_true", default=False, help='2D PES calculation')
    f.add_option('--natoms', type = int, default = None, help='Number of atoms in molecule')
    f.add_option('--log' , type = str, default = None, help='OpenMolcas log file')
    f.add_option('--atm1' , type = str, default = None, help='Atom 1 for 1D potential ')
    f.add_option('--job' , type = str, default = None, help='Job file name ')
    f.add_option('--oldname' , action="store_true", default=False, help='Old naming for Input files ')
    f.add_option('--oldinput' , action="store_true", default=False, help='Old input writting style (To be used with Rydberg basis) ')
    f.add_option('--pes' , type = str, default = None, help='PES file name ')
    
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

                        if arg.job:
                            job = arg.job
                        else:
                            job = file

                        if arg.oldname == True or arg.N > 99:
                            input=OldInputName(job,i,j)
                        else:
                            input=InputName(job,i,j)

                        WriteInput(input, arg.bs, atomnames, TdXYZ, arg.sym, Q[i], Q2[j])
                        #if arg.oldinput == True:
                        #    WriteInput(input, arg.bs, atomnames, TdXYZ, arg.sym, Q[i], Q2[j])
                        #else:
                        #    WriteXYZ(input, atomnames, TdXYZ, Q[i], Q2[j])
                
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
                    if arg.job:
                        job = arg.job
                    else:
                        job = file

                    if arg.oldname == True or arg.N > 99:
                        input=OldInputName(job,i,None)
                    else:
                        input=InputName(job,i,None)


                    WriteInput(input, arg.bs, atomnames, NewXYZ, arg.sym, Q[i], None)
                    #if arg.oldinput == True:
                    #    WriteInput(input, arg.bs, atomnames, NewXYZ, arg.sym, Q[i], None)
                    #else:
                    #    WriteXYZ(input, atomnames, NewXYZ, Q[i], None)

                
                    job=input.replace(".input", ".sh")
                    scpt=open(job, 'w')
                
                    with open("submit.sh") as fugu:
                        scpt.write(fugu.read())
                    scpt.write("pymolcas -b 1 -f %s" % input)
                    fugu.close()
                    scpt.close()
                    
                    #call("sbatch %s" % job, shell=True)
        



            
    elif arg.diff == True:

        
        eqxyz,atomnames=readxyz(arg.eq)
        #nmxyz,test=readxyz(arg.nm)
        refxyz,dump=readxyz(arg.ref)

        
        diff=eqxyz-refxyz

        title = "Difference between " + arg.eq + " and " + arg.ref
        
        #if arg.diff == True:
        print_coord(atomnames,diff,title)
        
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

            q1,q2=get_Q_2D()
            #Q1=np.linspace(arg.qi,arg.qf,arg.N)
            #Q2=np.linspace(arg.qi2,arg.qf2,arg.N)
            #q1=[]
            #q2=[]
            #for i in range(arg.N):
            #    for j in range(arg.N):
            #        q1.append(Q1[i])
            #        q2.append(Q2[j])
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

        #Q1, Q2 = get_Q_2D()

        #Q1=np.linspace(arg.qi,arg.qf,arg.N)
        #Q2=np.linspace(arg.qi2,arg.qf2,arg.N)

        files=sorted(glob.iglob('*.log'))
        i=0
        j=0
        J=0
        dgdq=2.41719
        for file in files:
            symb,geoxyz,nacxyz,fcsf,energy,norm=get_nac(file, arg.natoms)

            Q1, Q2 = get_2D_Q(file)

            #print(geoxyz)
            
            dq=0.00001
            Gdq=generate_coord(geoxyz, qxyz, dq)
            dq2=-0.00001
            Gdq2=generate_coord(geoxyz, qxyz, dq2)
            


            dgdq=(Gdq-geoxyz)/dq
            dgdq2=(np.array(Gdq)-np.array(Gdq2))/(2*dq)
            
            #nacme=np.sum(nacxyz)
            nacme2=np.sum(dgdq*nacxyz)
            

            #dd=dgdq*nacxyz
            #print(dgdq)
            
            ##nacme=np.sum(nacxyz)
            #h=(nacxyz-fcsf)*energy
            #nacme=np.sum(dgdq*h)

            
            print ("%.6s" % Q1[0], end=' ')
            print ("%.6s" % Q2[0], end='  ')
            
            print(abs(nacme2), end='  ')
            #print(abs(np.sum(h)), end=' ')
            #print("%.8E" % abs(energy), end=' ')
            #print(nacxyz, end=' ')
            #print(abs(1/np.sum(nacxyz)), end=' ')
            print(' ')

            i=i+1
            if i == arg.N:
                i=0
                print(' ')

    # GET NACME FROM OPENMOLCAS 
    elif arg.getnac1d == True:
        
        natoms=2
        R,SNAC,hAB=get_nac1D(arg.log,natoms)

        for i in range(len(R)):
            print(R[i], abs(1/SNAC[i]))
            #print(R[i], abs(hAB[i]))
            
    # GET NACMEs AND PEC FROM MOLPRO OUTPUT FOR DIATOMIC MOLECULES
    elif arg.getnacme == True:
        natoms=2
        R, NACME, E1, E2 = get_nacme_molpro(arg.log, natoms)

        #print("# MOLPRO NACME")
        for i in range(len(R)):
            print(R[i], abs(NACME[i]))
            
        #print("# MOLPRO PECs")
        #for i in range(len(R)):
        #    print(R[i], E1[i], E2[i])


    # GET QDV RESULTS FROM OPENMOLCAS 
    elif arg.getdqv  == True:
        
        if arg.log:
            log=arg.log
            natoms=2
            nD="diatomic"
            R, Q1, Q2, E, RM, W = get_dqv(log, natoms,nD)

            OSL, OSV, TDM, DX, DY, DZ=get_dipole(log)
            
            MTDM= np.array([[0,TDM],[TDM,0]])
            IRM=np.linalg.inv(RM)
            DTDM1=MTDM*IRM
            DTDM=DTDM1*RM
            
            MDX= np.array([[0,DX],[DX,0]])
            DDX1=MDX*IRM
            DDX=DDX1*RM
            
            MDY= np.array([[0,DY],[DY,0]])
            DDY1=MDY*IRM
            DDY=DDY1*RM

            MDZ= np.array([[0,DZ],[DZ,0]])
            DDZ1=MDZ*IRM
            DDZ=DDZ1*RM

            NEWTDM=math.sqrt(DDX[0][1]**2+DDY[0][1]**2+DDZ[0][1]**2)
            

            
            print(RM)
            print(abs(DTDM[0][1]))
            print(NEWTDM)


            #dipole=get_dipole(log)
            
            pot=log.replace(".log", "-diab.pes.dat")
            out=open(pot,"w")
            out.write("# QDV potentials \n")
            for i in range(len(R)):
                out.write("%.4s %.16s %.16s \n" % (R[i], E[i][0], E[i][3]))
            out.close()
            
            coup=log.replace(".log", "-diab.coup.dat")
            out=open(coup,"w")
            out.write("# QDV Couplings elements U12 \n")    
            for i in range(len(R)):   
                #out.write("%.4s %.16s \n" % (R[i], DC[i][1]**2 ))
                out.write("%.4s %.16E \n" % (R[i], E[i][1]**2 ))
            out.close()

            #dip=log.replace(".log", "-dipoles.dat")
            #out=open(dip,"w")
            #out.write("# Transitions dipole moment \n")    
            #for i in range(len(R)):   
            #    out.write("%.4s %.16s \n" % (R[i], dipole[i]))
            #out.close()           
            
        elif arg.td:
            nD = 2
            natoms = 2
            files=sorted(glob.iglob('*.log'))
            q1=0
            
            gspot=open("gs-surf.dat", "w")
            s1pot=open("s1-surf.dat", "w")
            dcpot=open("dc-surf.dat", "w")
            tdmpot=open("tdm-surf.dat", "w")
            
            for log in files:
                R,  Q1, Q2, E, RM, W = get_dqv(log, natoms, nD)
                OSL, OSV, TDM, DX, DY, DZ=get_dipole(log)

                MTDM= np.array([[0,TDM],[TDM,0]])
                IRM=np.linalg.inv(RM)
                DTDM1=MTDM*IRM
                DTDM=DTDM1*RM
                
                if E.any():

                    if Q1[0] != q1:
                        gspot.write('\n')
                        s1pot.write('\n')
                        dcpot.write('\n')
                        tdmpot.write('\n')
                        
                    gspot.write("%.8s %.8s %.16s \n" %  (Q1[0], Q2[0], E[0][0]  ))
                    s1pot.write("%.8s %.8s %.16s \n" %  (Q1[0], Q2[0], E[0][3]  ))
                    dcpot.write("%.8s %.8s %.6E \n " %  (Q1[0], Q2[0], abs(1/E[0][1]) ))
                    tdmpot.write("%.8s %.8s %.8E \n" % (Q1[0], Q2[0], abs(DTDM[0][1])  ))
                    
                    q1=Q1[0]
            gspot.close()
            s1pot.close()
            dcpot.close()
            tdmpot.close()
            
        else:
            nD = 1
            natoms = 2
            files=sorted(glob.iglob('*.log'))
            
            gspot=open("gs-surf.dat", "w")
            s1pot=open("s1-surf.dat", "w")
            dcpot=open("dc-surf.dat", "w")
            tdmpot=open("tdm-surf.dat", "w")
            
            for log in files:
                R,  Q, Q2, E, RM, W = get_dqv(log, natoms, nD)
                OSL, OSV, TDM, DX, DY, DZ=get_dipole(log)
                #TDM=OSV
                MTDM= np.array([[0,OSV],[OSV,0]])
                IRM=np.linalg.inv(RM)
                DTDM1=MTDM*IRM
                DTDM=DTDM1*RM
                
                if E.any():
                    gspot.write("%.8s  %.16s \n" % (Q[0], E[0][0]) )
                    s1pot.write("%.8s  %.16s \n" % (Q[0], E[0][3]) )
                    dcpot.write("%.8s  %.6E \n " % (Q[0], 1/E[0][1]**2 ) )

                    tdmpot.write("%.8s %.16s \n" % (Q[0], abs(DTDM[0][1]))  )
                    #tdmpot.write("%.8s %.16s \n" % (Q[0], OSV )  )
                    
            gspot.close()
            s1pot.close()
            dcpot.close()
            tdmpot.close()
                    
    elif arg.getdm == True:
        nD = 2
        natoms = 2
        files=sorted(glob.iglob('*.log'))
        qmark=0

        osclen=open("osc-len-surf.dat", "w")
        oscvel=open("osc-vel-surf.dat", "w")
        tdmlen=open("tdm-len-surf.dat", "w")
        tdmvel=open("tdm-vel-surf.dat", "w")

        for log in files:

            # Get Q
            if arg.td == True:
                Q, Q2 = get_2D_Q(log)
            else:
                Q = get_Q_1D(log)

            # Get Oscillator/TDM
            OSL, OSV, TDM, DX, DY, DZ, E = get_dipole(log)

            #energy=get_energy("RASSI", "2")

            # Following OpenMolcas manual (Sec. 5.1.5.1.5, p.587)
            ediff=abs(E[1]-E[0])
            VTDM = math.sqrt( 1.5 * (OSV/ediff) )
           
            # Write 2D surfaces
            if arg.td == True:
                if Q[0] != q1:
                    osclen.write('\n')
                    oscvel.write('\n')
                    tdmlen.write('\n')
                    tdmvel.write('\n')
                    
                osclen.write("%.8s  %.8s  %.16s \n" % (Q[0], Q2[0], OSL) )
                oscvel.write("%.8s  %.8s  %.16s \n" % (Q[0], Q2[0], OSV) )
                tdmlen.write("%.8s  %.8s  %.16s \n" % (Q[0], Q2[0], TDM) )
                tdmvel.write("%.8s  %.8s  %.16s \n" % (Q[0], Q2[0], VTDM) )
                q1=Q[0]
                
            # Write 1D surfaces
            else:
                osclen.write("%.8s %.16s \n" % (Q[0], OSL) )
                oscvel.write("%.8s %.16s \n" % (Q[0], OSV) )
                tdmlen.write("%.8s %.16s \n" % (Q[0], TDM) )
                tdmvel.write("%.8s %.16s \n" % (Q[0], VTDM) )
            
        osclen.close()
        oscvel.close()
        tdmlen.close()
        tdmvel.close()

    ########################
    # CALCULATE THE G-MATRIX
    elif arg.gmatrix == True:

        # Read Q1 and Q2 files
        q1,symb=readxyz(arg.nm)
        q2,dump=readxyz(arg.nm2)
        
        #Name the output files
        g11out=open("g11-au.dat", "w")
        g12out=open("g12-au.dat", "w")
        g22out=open("g22-au.dat", "w")

        # Marker
        qmark=0.0

        # Loop over all .log files
        files=sorted(glob.iglob('*.log'))
        for log in files:
            
            # Get geometry from log file
            geo, Q1, Q2 = get_geo(log)

            # Compute the centrer of mass of geometry
            COM, MWgeo, M = center_of_mass(symb, geo)

            # Move geometry, Q1 and Q2 to center of mass
            geoCOM = geo - COM
            q1COM  = q1 - COM
            q2COM  = q2 - COM

            ### DERIVATIVES
            # Foward
            dQf = 1e-9
            # Backward
            dQb = -1e-9
            
            ### Set up for Q1 derivatives
            # generate sligthly distorted molecule
            dXdQ1f=generate_coord(geoCOM, q1COM, dQf)
            dXdQ1b=generate_coord(geoCOM, q1COM, dQb)
            # Align structures
            dXdQ1f, rmsd = align_struct(np.array(geoCOM), np.array(dXdQ1f))
            dXdQ1b, rmsd = align_struct(np.array(geoCOM), np.array(dXdQ1b))
            
            ### Set up for Q2 derivatives
            # generate sligthly distorted molecule
            dXdQ2f=generate_coord(geoCOM, q2COM, dQf)
            dXdQ2b=generate_coord(geoCOM, q2COM, dQb)
            # Align structures
            dXdQ2f, rmsd = align_struct(np.array(geoCOM), np.array(dXdQ2f))
            dXdQ2b, rmsd = align_struct(np.array(geoCOM), np.array(dXdQ2b))

            # Convert coordinates to Bohr
            geoCOM = geoCOM*1.889725989
            dXdQ1f = dXdQ1f*1.889725989
            dXdQ1b = dXdQ1b*1.889725989
            dXdQ2f = dXdQ2f*1.889725989
            dXdQ2b = dXdQ2b*1.889725989
            dQf = dQf*1.889725989
            dQb = dQb*1.889725989

            ## Q1 Derivatives
            # Foward diference derivative
            FdXdQ1=(dXdQ1f-geoCOM)/dQf
            # Central diference derivative
            CdXdQ1=(np.array(dXdQ1f)-np.array(dXdQ1b))/(2*dQf)

            ## Q2
            # Foward diference derivative
            FdXdQ2=(dXdQ2f-geoCOM)/dQf
            # Central diference derivative
            CdXdQ2=(np.array(dXdQ2f)-np.array(dXdQ2b))/(2*dQf)
            
            # Compute the G-matrix elements
            G11 = np.sum(M*CdXdQ1*CdXdQ1)
            G12 = np.sum(M*CdXdQ1*CdXdQ2)
            G22 = np.sum(M*CdXdQ2*CdXdQ2)
            
            # Build the matrix G_{rs}
            gmatrix=np.array([[G11,G12],[G12,G22]])
            
            # Invert the matrix to get G^{rs}
            Igmatrix=np.linalg.inv(gmatrix)
            
            # Print G-matrix to file
            if Q1[0] != qmark:
                g11out.write('\n')
                g12out.write('\n')
                g22out.write('\n')
            g11out.write("%.8s  %.8s  %.16E \n" % (Q1[0], Q2[0], Igmatrix[0][0]) )
            g12out.write("%.8s  %.8s  %.16E \n" % (Q1[0], Q2[0], Igmatrix[1][0]) )
            g22out.write("%.8s  %.8s  %.16E \n" % (Q1[0], Q2[0], Igmatrix[1][1]) )

            qmark = Q1[0]

        g11out.close()
        g12out.close()
        g22out.close()

        
    # Shift 2D PES to minimum
    elif arg.shiftpes == True:

        surf=arg.pes
        
        pes = readpes(surf)
        
        Q1 = pes[:,0]
        Q2 = pes[:,1]
        E  = pes[:,2]
  
        Emin = min(float(s) for s in E)

        
        shifted=surf.replace(".pot", "-shifted.pot")
        shifted2=surf.replace(".pot", "-shifted-Z.pot")
        z=open(shifted2, "w")
        out=open(shifted, "w")

        qmark = 0.0
        for i in range(len(E)):
            if Q1[i] != qmark:
                out.write('\n')
            #print(E[i]-Emin)
            out.write("%.8s  %.8s  %.16E \n" % (Q1[i], Q2[i], E[i]-Emin ) )
            z.write(" %.16E \n" % ( E[i]-Emin ) )
            qmark = Q1[i] 


        
            

    elif arg.plotpes == True:

        surf=arg.pes
        
        pes = readpes(surf)
        
        Q1 = pes[:,0]
        Q2 = pes[:,1]
        E  = pes[:,2]
  

        
        shifted=surf.replace(".pot", "-fixed.pot")
        out=open(shifted, "w")

        qmark = 0.0
        for i in range(len(E)):
            if Q2[i] != qmark:
                out.write('\n')
            out.write("%.8s  %.8s  %.16E \n" % (Q2[i], Q1[i], E[i] ) )
            qmark = Q2[i] 

        
if __name__=="__main__":
    main()
