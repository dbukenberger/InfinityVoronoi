import sys
import os
import copy
import hashlib
import logging
import warnings
import multiprocessing as mp
from time import time
from xml.dom import minidom
from abc import ABC, abstractmethod
import numpy as np
np.random.seed(23)


# log and result directories
logDir = 'logs/'
resDir = 'results/'
for pDir in [logDir, resDir]:
    if not os.path.exists(pDir):
        os.mkdir(pDir)


# useful global constants
IN_IDLE = 'idlelib.run' in sys.modules
cpuCount = int(sys.argv[-1]) if '-t' in sys.argv else mp.cpu_count()
eps = 0.000001
dotEps = np.deg2rad(0.1)
quadVerts = np.float32([[-1,-1],[-1,1],[1,1],[1,-1]])
cubeVerts = np.float32([[-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1],[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]])
sixCubeFaces = np.int32([[0,1,2,3],[4,0,3,7],[5,1,0,4],[7,3,2,6],[1,5,6,2],[5,4,7,6]])


# for visualizing 2D results
try:
    import matplotlib.pyplot as plt
except ImportError:
    mplMissing = True
else:
    mplMissing = False


# for visualizing 3D results
try:
    from mayavi import mlab
except ImportError:
    mlabMissing = True
else:
    mlabMissing = False


# show progress in shell
try:
    if IN_IDLE:
        raise RuntimeError
    from tqdm import tqdm
except:

    class tqdmDummy:
        n = 0

        def update(self, x):
            return

        def close(self):
            return

    def tqdm(x, **kwargs):
        return tqdmDummy() if x is None else x


# simple logger class
logging.basicConfig(format='%(message)s')
class Logger:
    def __init__(self, logName):
        self.log = logging.getLogger(logName)
        self.log.setLevel(logging.INFO)
        self.log.addHandler(logging.FileHandler(logDir + '%s.log' % logName, mode='w'))

    def logThis(self, msg, args, style=None):
        tplArgs = args if not hasattr(args, '__len__') or type(args) == str else tuple(args)
        fmtArgs = style % tplArgs if style is not None else str(tplArgs)
        self.log.info(msg + ':\t' + fmtArgs)


# basic geometry and utility functions

try:
    from numpy.core.umath_tests import inner1d
except ImportError:
    def inner1d(u, v): return np.einsum('ij,ij->i', u, v)

def normVec(v):
    if v.ndim == 1:
        n = np.sqrt(np.dot(v, v))
        return v / n if n else v * 0
    else:
        n = np.sqrt(inner1d(v, v))
        m = n != 0
        v = v.copy()
        v[m] /= n[m].reshape(-1, 1)
        return v

def norm(v): return np.sqrt(np.dot(v, v) if v.ndim == 1 else inner1d(v, v))

def orthoVec(v): return [1, -1] * (v[::-1] if v.ndim == 1 else v[:, ::-1])

def randomJitter(n, k, s=1): return normVec(np.random.rand(n, k) * 2 - 1) * np.random.rand(n, 1) * s

def generateJitteredGridPoints(n, d, e=1): return generateGridPoints(n, d, e) + randomJitter(n**d, d, e / n)

def vecsParallel(u, v, signed=False): return 1 - np.dot(u, v) < eps if signed else 1 - np.abs(np.dot(u, v)) < eps

def distPointToPlane(p, o, n): return np.abs(np.dot(p - o, n))

def planesEquiv(onA, onB): return distPointToPlane(onA[0], onB[0], onB[1]) < eps and vecsParallel(onA[1], onB[1])

def innerNxM(Ns, Ms): return np.einsum('ijh,ikh->ijk', Ns, Ms) # np.vstack([np.dot(N, M.T) for N, M in zip(Ns, Ms)])

def Mr2D(a): return np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

def Mr(ori): return Mr2D(ori) if np.isscalar(ori) else Mr3D(ori[0], ori[1], ori[2])

def padHor(pts, c=0): return np.pad(pts, [[0,0],[0,1]], mode='constant', constant_values=c)

def toEdgeTris(es): return np.pad(es, [[0,0],[0,1]], mode='reflect')

def flatten(lists): return [element for elements in lists for element in elements]

def rgb2hex(rgbCol): return '#'+''.join([hex(int(c))[2:].zfill(2) for c in rgbCol])
def hex2rgb(hexCol): return [int(cv * (1 + (len(hexCol) < 5)), 16) for cv in map(''.join, zip(*[iter(hexCol.replace('#', ''))] * (2 - (len(hexCol) < 5))))]
def seed2hex(seed): return '#' + hashlib.md5(str(seed).encode()).hexdigest()[-6:]
def seed2rgb(seed): return hex2rgb(seed2hex(seed))
def rgb2rgba(rgbCol, alpha=255): return padHor(rgbCol, alpha)

def cantorPi(k1, k2): return ((k1 + k2) * (k1 + k2 + 1)) // 2 + (k1 if k1 > k2 else k2)
def cantorPiO(k1, k2): return ((k1 + k2) * (k1 + k2 + 1)) // 2 + k2


def cantorPiV(k1k2):
    if k1k2.dtype != np.int64:
        k1k2 = np.int64(k1k2)
    k1k2.sort(axis=1)
    k1k2sum = k1k2[:, 0] + k1k2[:, 1]
    return ((k1k2sum * k1k2sum + k1k2sum) >> 1) + k1k2[:, 1]


def simpleSign(xs, thresh=None):
    signs = np.int32(np.sign(xs))
    if thresh is None:
        return signs
    else:
        return signs * (np.abs(xs) > thresh)


def cross(a, b, normed=False):
    if a.ndim == 1 and b.ndim == 1:
        c = np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
    if a.ndim == 1 and b.ndim > 1:
        c = np.array([a[1]*b[:,2]-a[2]*b[:,1],a[2]*b[:,0]-a[0]*b[:,2],a[0]*b[:,1]-a[1]*b[:,0]]).T
        #c = np.array([b[:,1]*a[2]-b[:,2]*a[1],b[:,2]*a[0]-b[:,0]*a[2],b[:,0]*a[1]-b[:,1]*a[0]]).T
    if a.ndim > 1 and b.ndim == 1:
        return cross(b, a, normed)
    if a.ndim > 1 and b.ndim > 1:
        c = np.array([a[:,1]*b[:,2]-a[:,2]*b[:,1],a[:,2]*b[:,0]-a[:,0]*b[:,2],a[:,0]*b[:,1]-a[:,1]*b[:,0]]).T
    return normVec(c) if normed else c


def Mr3D(alpha=0, beta=0, gamma=0):
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, -np.sin(beta)], [0, 1, 0], [np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))


def generateGridPoints(n, d, e=1):
    ptsGrid = np.linspace(-e, e, n, endpoint=False) + e / n
    if d == 1:
        return ptsGrid
    if d == 2:
        return np.vstack(np.dstack(np.meshgrid(ptsGrid, ptsGrid)))
    elif d == 3:
        return np.float32(np.vstack(np.vstack(np.transpose(np.meshgrid(ptsGrid, ptsGrid, ptsGrid), axes=[3,2,1,0]))))
    else:
        warnings.warn('%d dimensions not supported'%d)
        return


def distPointToEdge(A, B, P):
    AtoB = B - A
    AtoP = P - A
    BtoP = P - B
    
    if np.dot(AtoB, AtoP) <= 0:
        return norm(AtoP)
    elif np.dot(-AtoB, BtoP) <= 0:
        return norm(BtoP)
    else:
        d = normVec(AtoB)
        return norm(AtoP-np.dot(AtoP,d)*d)


def edgesIntersect2D(A, B, C, D):
    d = (D[1] - C[1]) * (B[0] - A[0]) - (D[0] - C[0]) * (B[1] - A[1])
    u = (D[0] - C[0]) * (A[1] - C[1]) - (D[1] - C[1]) * (A[0] - C[0])
    v = (B[0] - A[0]) * (A[1] - C[1]) - (B[1] - A[1]) * (A[0] - C[0])
    if d < 0:
        u, v, d = -u, -v, -d
    return (0 <= u <= d) and (0 <= v <= d)


def pointInTriangle2D(A, B, C, P):
    v0, v1, v2 = C - A, B - A, P - A
    u, v, d = v2[1] * v0[0] - v2[0] * v0[1], v1[1] * v2[0] - v1[0] * v2[1], v1[1] * v0[0] - v1[0] * v0[1]
    if d < 0:
        u, v, d = -u, -v, -d
    return u >= 0 and v >= 0 and (u + v) <= d


def pointInTriangles3D(ABCs, P, uvws = None, ns = None, assumeInPlane = False):
    if uvws is None:
        uvws = ABCs[:,[1,2,0]] - ABCs
    if assumeInPlane:
        m = np.ones(len(ABCs), np.bool8)
    else:
        m = np.abs(inner1d(P-ABCs[:,0], cross(uvws[:,0], uvws[:,1], True) if ns is None else ns[:,0])) < eps
        if not m.any():
            return False
    for i in range(3):
        m *= inner1d(cross(uvws[:,i], -uvws[:,(i+2)%3], True) if ns is None else ns[:,i], cross(uvws[:,i], P-ABCs[:,i], True)) > 0
        if not m.any():
            return False
    return True


def trianglesDoIntersect2D(t1, t2=None):
    if t2 is None:
        t1, t2 = t1
    for i in range(3):
        for j in range(3):
            if edgesIntersect2D(t1[i], t1[(i+1)%3], t2[j], t2[(j+1)%3]):
                return True
    for p in t2:
        if not pointInTriangle2D(t1[0], t1[1], t1[2], p):
            break
    else:
        return True
    for p in t1:
        if not pointInTriangle2D(t2[0], t2[1], t2[2], p):
            break
    else:
        return True
    return False


def pyrasDoIntersect(ptsA, ptsB, nsA, nsB):
    masks = np.zeros((5,5), np.bool8)
    coord = np.zeros((5,5), np.float32)

    def faceA(i):
        # ptsA[i] only works with correct ns order!
        coord[i] = np.dot(ptsB - ptsA[i], nsA[i])
        masks[i] = coord[i] > 0
        return np.all(masks[i])

    def faceB(i):
        return np.all(np.dot(ptsA - ptsB[i], nsB[i]) > 0)

    def edge(f, g):
        if not np.all(np.bitwise_or(masks[f], masks[g])):
            return False

        for e in [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]:
            if (masks[f,e[0]] and not masks[f,e[1]]) and (not masks[g,e[0]] and masks[g,e[1]]):
                if (coord[f,e[1]] * coord[g,e[0]] - coord[f,e[0]] * coord[g,e[1]]) > 0:
                    return False
            if (masks[f,e[1]] and not masks[f,e[0]]) and (not masks[g,e[1]] and masks[g,e[0]]):
                if (coord[f,e[1]] * coord[g,e[0]] - coord[f,e[0]] * coord[g,e[1]]) < 0:
                    return False

        return True

    def pointInside():
        return not np.all(np.any(masks, axis=0))

    fs = []
    for f in [[0],[1],[0,1],[2],[1,2],[3],[2,3],[3,0],[4],[0,4],[1,4],[2,4],[3,4]]:
        if len(f) == 1:
            if faceA(f[0]):
                return False
        else:
            if edge(f[0],f[1]):
                return False

    if pointInside():
        return True

    for f in range(5):
        if faceB(f):
            return False

    return True


def intersectLinesLine2D(ons, o, n):
    ss = np.dot(ons[:,0] - o, n) / np.dot(ons[:,1], orthoVec(n))
    return ons[:,0] + orthoVec(ons[:,1]) * ss.reshape(-1,1)


def intersectEdgePlane(pts, o, n, sane=True):
    if not sane:
        ds = np.dot(pts - o, n)
        if np.all(ds < 0) or np.all(ds > 0) or np.any(ds == 0):
            return None
    v = normVec(pts[1] - pts[0])
    t = np.dot((o-pts[0]), n) / np.dot(v, n)
    return pts[0] + v * t


def intersectEdgesPlane(pts, o, n):
    vs = pts[:,1] - pts[:,0]
    ts = np.dot((o-pts[:,0]), n) / np.dot(vs, n)
    return pts[:,0] + vs * ts.reshape(-1,1)


def intersectThreePlanes(os, ns):
    return np.linalg.solve(ns, inner1d(os, ns)) #if np.linalg.det(ns) else np.linalg.lstsq(ns, inner1d(os,ns))[0]


def faceToEdges(face):
    face = np.int32(face) if type(face) == list else face
    return face[np.roll(np.repeat(range(len(face)),2), -1)].reshape(-1,2)


def facesToEdges(faces):
    es = np.vstack([faceToEdges(face) for face in faces])
    return filterForUniqueEdges(es)


def facesToTris(faces):
    if type(faces) == list:
        fLens = list(map(len, faces))
        maxLen = max(fLens)
        mask = np.arange(maxLen) < np.array(fLens)[:,None]
        fcs = np.zeros((len(faces), maxLen), np.int32) - 1
        fcs[mask] = np.concatenate(faces)
        faces = fcs

    tris = np.hstack([np.repeat(faces[:,0].reshape(-1,1), faces.shape[1] - 2, axis=0), np.repeat(faces[:,1:],2, axis=1)[:,1:-1].reshape(-1,2)])
    return tris[np.bitwise_and(tris[:,1]>=0 , tris[:,2]>=0)]


def triangulatePoly2D(vs):
    tris = []
    poly = list(range(len(vs)))

    # check winding and flip for CW order
    if 0 > np.prod(vs * [[-1,1]] + np.roll(vs, -1, axis=0), axis=1).sum():
        poly = poly[::-1]

    idx = 0
    while len(poly) > 2:
        pdx, ndx = (idx-1)%len(poly), (idx+1)%len(poly)
        A, B, C = vs[poly[pdx]], vs[poly[idx]], vs[poly[ndx]]

        # check if concave or convex triangle
        if 0 < np.sign((B[0]-A[0]) * (C[1]-A[1]) - (B[1]-A[1]) * (C[0]-A[0])):
            idx = (idx+1)%len(poly)
            continue

        otherIdxs = [i for i in poly if i not in [poly[pdx], poly[idx], poly[ndx]]]
        for odx in otherIdxs:
            if pointInTriangle2D(A, B, C, vs[odx]):
                idx = (idx+1)%len(poly)
                break
        else:
            tris.append([poly[pdx], poly[idx], poly[ndx]])
            poly.pop(idx)
            idx %= len(poly)

    return np.int32(tris)


def triangulatePoly3D(vs):
    return triangulatePoly2D(aaProject3Dto2D(vs))


def aaProject3Dto2D(verts):
    vecs = normVec(verts - verts.mean(axis=0))
    eVals, eVecs = np.linalg.eig(np.dot(vecs.T, vecs))
    pDim = np.abs(eVecs[:,eVals.argmin()]).argmax()
    pDir = np.eye(3)[pDim]
    pVerts = verts - pDir * np.dot(verts, pDir).reshape(-1,1)
    return pVerts[:, np.int32([pDim + 1, pDim + 2]) % 3]


def edgesToPath(edgesIn):
    edges = copy.deepcopy(edgesIn) if type(edgesIn) == list else edgesIn.tolist()
    nLim = np.arange(len(edges) - 1).sum()
    nTries = 0
    face = edges.pop(0)
    while len(edges):
        edge = edges.pop(0)
        if face[0] == edge[0]:
            face.insert(0, edge[1])
        elif face[-1] == edge[0]:
            face.append(edge[1])
        elif face[0] == edge[1]:
            face.insert(0, edge[0])
        elif face[-1] == edge[1]:
            face.append(edge[0])
        else:
            edges.append(edge)
            nTries += 1
        if nTries > nLim:
            return
    return face if face[0] != face[-1] else face[:-1]


def edgesToPaths(edgesIn):
    edges = copy.deepcopy(edgesIn) if type(edgesIn) == list else edgesIn.tolist()
    face = edges.pop(0)
    faces = []
    iters = 0
    while len(edges):
        edge = edges.pop(0)
        if face[0] == edge[0]:
            face.insert(0, edge[1])
            iters = 0
        elif face[-1] == edge[0]:
            face.append(edge[1])
            iters = 0
        elif face[0] == edge[1]:
            face.insert(0, edge[0])
            iters = 0
        elif face[-1] == edge[1]:
            face.append(edge[0])
            iters = 0
        else:
            edges.append(edge)
            iters += 1
        if len(edges) and iters >= len(edges):
            iters = 0
            faces.append(copy.deepcopy(face if face[0] != face[-1] else face[:-1]))
            face = edges.pop(0)
    faces.append(face if face[0] != face[-1] else face[:-1])
    return faces


def findConnectedComponents(edges):
    comps = [set(edges[0])]
    for edge in edges[1:]:
        cIdxs = [cIdx for cIdx, comp in enumerate(comps) if not comp.isdisjoint(edge)]
        if not len(cIdxs):
            comps.append(set(edge))
        elif len(cIdxs) == 1:
            comps[cIdxs[0]].update(edge)
        elif cIdxs[0] != cIdxs[1]:
            comps[cIdxs[0]].update(comps.pop(cIdxs[1]))
    return comps


def findConnectedEdgeSegments(edges):
    segments = [[edge.tolist() if not type(edge) == list else edge] for edge in edges]
    while True:
        l = len(segments)
        for i, segmentA in enumerate(segments):
            for j, segmentB in enumerate(segments):
                if i == j:
                    continue
                if not set(flatten(segmentA)).isdisjoint(set(flatten(segmentB))):
                    segments[j] += segments[i]
                    segments.pop(i)
                    break
        if l == len(segments):
            break
    return segments


def appendUnique(lst, i):
    if i not in lst:
        lst.append(i)


def haveCommonElement(a, b):
    if len(b) < len(a):
        a, b = b, a
    for x in a:
        if x in b:
            return True
    return False


def filterForUniqueEdges(edges):
    edges = np.int32(edges) if type(edges) == list else edges
    uHashs, uIdxs = np.unique(cantorPiV(edges), return_index=True)
    return edges[uIdxs]


def filterForSingleEdges(edges):
    hashs = cantorPiV(edges)
    uHashs, uIdxs, uCounts = np.unique(hashs, return_index=True, return_counts=True)
    return np.array([edges[idx] for idx, count in zip(uIdxs, uCounts) if count == 1])


def reIndexIndices(arr):
    uIdxs = np.unique(flatten(arr))
    reIdx = np.zeros(uIdxs.max()+1, np.int32)
    reIdx[uIdxs] = np.argsort(uIdxs)
    return [reIdx[ar] for ar in arr] if type(arr) == list else reIdx[arr]


def computePolygonCentroid2D(pts, withArea=False):
    rPts = np.roll(pts, 1, axis=0)
    w = pts[:,0] * rPts[:,1] - rPts[:,0] * pts[:,1]
    area = w.sum() / 2.0
    centroid = np.sum((pts + rPts) * w.reshape(-1,1), axis=0) / (6 * area)
    return (centroid, np.abs(area)) if withArea else centroid


def computeTetraVolume(pts):
    a, b, c = pts[1:] - pts[0]
    return np.abs(np.dot(cross(a,b), c) / 6.0)


def computeTetraVolumes(ptss):
    abcs = ptss[:,1:] - ptss[:,0].reshape(-1,1,3)
    return np.abs(inner1d(cross(abcs[:,0], abcs[:,1]), abcs[:,2]) / 6.0)


def computePolyVolume(pts, faces):
    tris = facesToTris(faces) # works with convex faces only
    center = pts[np.unique(tris)].mean(axis=0)
    return np.sum([computeTetraVolume(np.vstack([center, pts[tri]])) for tri in tris])


def computePolyhedronCentroid(vertices, faces, returnVolume=False):
    tris = facesToTris(faces)
    tets = padHor(tris, -1)
    verts = np.vstack([vertices, [vertices[np.unique(tris)].mean(axis=0)]])
    tetPts = verts[tets]
    tetCentroids = tetPts.mean(axis=1)
    tetVolumes = computeTetraVolumes(tetPts)
    tetVolumesSum = tetVolumes.sum()
    polyCentroid = np.dot(tetVolumes/tetVolumesSum, tetCentroids) if tetVolumesSum > eps else tetCentroids.mean(axis=0)
    return (polyCentroid, tetVolumesSum) if returnVolume else polyCentroid


def getConvexPolygonVertexOrder(pts, refPt=None):
    cPt = pts.mean(axis=0)
    if refPt is not None:
        n = normVec(refPt - cPt)
        pts = projectPoints(pts, cPt, n, True)
        cPt = pts.mean(axis=0)
    dirs = normVec(pts - cPt)
    return np.argsort((np.arctan2(dirs[:,0], dirs[:,1]) + 2 * np.pi) % (2 * np.pi))


def projectPoints(pts, o, n, return2d=False):
    vs = pts - o
    ds = np.dot(vs, n)
    projected = pts - ds.reshape(-1,1) * n
    if not return2d:
        return projected

    up = np.float32([0,0,1])
    x = cross(up, n, True)
    theta = np.arccos(np.dot(up, n))
    A = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    R = np.eye(3) + np.sin(theta) * A + (1-np.cos(theta)) * np.dot(A,A)
    return np.dot(R.T, projected.T).T[:,:2]


def concatPolyParts(polyParts):
    polys = []
    for part in polyParts:
        for i, poly in enumerate(polys):
            if norm(part[-1] - poly[0]) < eps:
                polys[i] = np.vstack([part, poly])
                break
            if norm(part[0] - poly[0]) < eps:
                polys[i] = np.vstack([part[::-1], poly])
                break
            if norm(part[0] - poly[-1]) < eps:
                polys[i] = np.vstack([poly, part])
                break
            if norm(part[-1] - poly[-1]) < eps:
                polys[i] = np.vstack([poly, part[::-1]])
                break
        else:
            polys.append(part)
    return polys


def limitedDissolve2D(verts):
    vIdxs = []
    n = len(verts)
    for vIdx in range(n):
        pIdx = (vIdx - 1) % n
        nIdx = (vIdx + 1) % n
        if vIdx == nIdx or norm(verts[vIdx] - verts[nIdx]) < eps:
            continue
        vecs = normVec(verts[[pIdx, nIdx]] - verts[vIdx])
        if np.abs(np.dot(vecs[0], vecs[1])) < (1 - eps):
            vIdxs.append(vIdx)
    return limitedDissolve2D(verts[vIdxs]) if len(vIdxs) < n else verts[vIdxs]


def getFaceCutIdxs(faceMasks):
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


def writeObjFile(filePath, vertices, faces, edges=[], comment='', subTags=[]):
    fh = open(filePath, 'w')
    fh.write('# obj export\n')
    if len(comment):
        fh.write('# %s\n'%comment)
    if not len(subTags):
        fh.write('o %s\n' % os.path.basename(filePath)[:-4])
        for vertex in vertices:
            fh.write('v %0.6f %0.6f %0.6f\n' % tuple(vertex))
        for face in faces:
            fh.write(('f' + ' %d' * len(face) + '\n') % tuple(face + 1))
        for edge in edges:
            fh.write('l %d %d\n' % tuple(edge + 1))
    else:
        vOffset = 0
        for subIdx, tag in enumerate(subTags):
            fh.write('o %s\n' % tag)
            for vertex in vertices[subIdx]:
                fh.write('v %0.6f %0.6f %0.6f\n' % tuple(vertex))
            if len(faces):
                for face in map(np.int32, faces[subIdx]):
                    fh.write(('f' + ' %d' * len(face) + '\n') % tuple(face + vOffset + 1))
            if len(edges):
                for edge in map(np.int32, edges[subIdx]):
                    fh.write('l %d %d\n' % tuple(edge + vOffset + 1))
            vOffset += len(vertices[subIdx])
    fh.close()