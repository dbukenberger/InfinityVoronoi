from xml.dom import minidom
from abc import ABC, abstractmethod

# very general stuff is now part of the drbutil module
try:
    from drbutil import *
except ImportError:
    import os
    import sys
    import requests
    utilDir = 'drbutil/'
    if not os.path.exists(utilDir):
        os.mkdir(utilDir)
        print('drbutil not found, downloading ...')
        for fName in ['__init__', '__version__', 'util', 'io']:
            r = requests.get('https://raw.githubusercontent.com/dbukenberger/drbutil/main/src/drbutil/%s.py'%fName)
            if r.status_code != 200:
                print('Something went wrong, try downloading/installing drbutil manually.')
            else:
                with open(utilDir + fName + '.py', "w+") as pyFile:
                    pyFile.write(r.text)
        print('done, starting now ...')
    sys.path.append(utilDir)
    from drbutil import *


def computeFaceCutIdxs(faceMasks):
    faceMasksCat = np.concatenate(faceMasks)
    faceLensCum = np.cumsum([0]+list(map(len, faceMasks)))
    firstLastIdxs = np.transpose([faceLensCum[:-1], faceLensCum[1:]-1])

    firstLasts = faceMasksCat[firstLastIdxs.ravel()].reshape(-1,2)
    trueStartIdxs = faceLensCum[np.where(firstLasts.sum(axis=1) == 1)[0]]

    firstLastIdxsFlat = firstLasts.ravel()[:-1]
    falseStartIdxs = faceLensCum[np.where(np.reshape(firstLastIdxsFlat[1:] ^ firstLastIdxsFlat[:-1], (-1,2))[:,1])[0]+1]

    inFaceIdxs = np.where(faceMasksCat[:-1] ^ faceMasksCat[1:])[0]+1

    resIdxs = np.int32(sorted(set(inFaceIdxs).difference(falseStartIdxs).union(trueStartIdxs)))
    return resIdxs.reshape(-1,2) - np.reshape(faceLensCum[:-1], (-1,1))