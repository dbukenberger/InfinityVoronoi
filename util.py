import sys
import os
import copy
import hashlib
import multiprocessing as mp
from time import time
from xml.dom import minidom
import logging
import warnings
import numpy as np


try:
    from numpy.core.umath_tests import inner1d
except ImportError:
    inner1d = lambda u, v: np.einsum("ij,ij->i", u, v)


# for plotting results
try:
    import matplotlib.pyplot as plt
except ImportError:
    mplMissing = True
else:
    mplMissing = False


# show progress in shell
try:
    if "idlelib.run" in sys.modules:
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
logging.basicConfig(format="%(message)s")
class Logger:
    def __init__(self, logName):
        self.log = logging.getLogger(logName)
        self.log.setLevel(logging.INFO)
        self.log.addHandler(logging.FileHandler(logDir + "%s.log" % logName, mode="w"))

    def logThis(self, msg, args, style=None):
        tplArgs = args if not hasattr(args, "__len__") or type(args) == str else tuple(args)
        fmtArgs = style % tplArgs if style is not None else str(tplArgs)
        self.log.info(msg + ":\t" + fmtArgs)


# log and result directories
logDir = "logs/"
resDir = "results/"
for pDir in [logDir, resDir]:
    if not os.path.exists(pDir):
        os.mkdir(pDir)


IN_IDLE = "idlelib.run" in sys.modules
cpuCount = mp.cpu_count()
np.random.seed(23)
eps = 0.000001


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


norm = lambda v: np.sqrt(np.dot(v, v) if v.ndim == 1 else inner1d(v, v))

orthoVec = lambda v: [1, -1] * (v[::-1] if v.ndim == 1 else v[:, ::-1])

randomJitter = lambda n, k, s=1: normVec(np.random.rand(n, k) * 2 - 1) * np.random.rand(n, 1) * s

vecsParallel = lambda u, v, signed=False: 1 - np.dot(u, v) < eps if signed else 1 - np.abs(np.dot(u, v)) < eps

innerNxM = lambda Ns, Ms: np.einsum("ijh,ikh->ijk", Ns, Ms) # np.vstack([np.dot(N, M.T) for N, M in zip(Ns, Ms)])

Mr2D = lambda a: np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

pad2Dto3D = lambda pts, c=0: np.pad(pts, [[0, 0], [0, 1]], mode="constant", constant_values=c)

flatten = lambda lists: [element for elements in lists for element in elements]

rgb2hex = lambda rgbCol: '#'+''.join([hex(int(c))[2:].zfill(2) for c in rgbCol])
hex2rgb = lambda hexCol: [int(cv * (1 + (len(hexCol) < 5)), 16) for cv in map("".join, zip(*[iter(hexCol.replace("#", ""))] * (2 - (len(hexCol) < 5))))]
seed2hex = lambda seed: "#" + hashlib.md5(str(seed).encode()).hexdigest()[-6:]
seed2rgb = lambda seed: hex2rgb(seed2hex(seed))

cantorPi = lambda k1, k2: ((k1 + k2) * (k1 + k2 + 1)) // 2 + (k1 if k1 > k2 else k2)
cantorPiO = lambda k1, k2: ((k1 + k2) * (k1 + k2 + 1)) // 2 + k2


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


def trianglesDoIntersect2D(t1, t2=None):
    if t2 is None:
        t1, t2 = t1
    if edgesIntersect2D(t1[0], t1[1], t2[0], t2[1]): return True
    if edgesIntersect2D(t1[0], t1[1], t2[0], t2[2]): return True
    if edgesIntersect2D(t1[0], t1[1], t2[1], t2[2]): return True
    if edgesIntersect2D(t1[0], t1[2], t2[0], t2[1]): return True
    if edgesIntersect2D(t1[0], t1[2], t2[0], t2[2]): return True
    if edgesIntersect2D(t1[0], t1[2], t2[1], t2[2]): return True
    if edgesIntersect2D(t1[1], t1[2], t2[0], t2[1]): return True
    if edgesIntersect2D(t1[1], t1[2], t2[0], t2[2]): return True
    if edgesIntersect2D(t1[1], t1[2], t2[1], t2[2]): return True
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


def intersectLinesLine2D(ons, o, n):
    ss = np.dot(ons[:, 0] - o, n) / np.dot(ons[:, 1], orthoVec(n))
    return ons[:, 0] + orthoVec(ons[:, 1]) * ss.reshape(-1, 1)


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


def filterForUniqueEdges(edges):
    edges = np.int32(edges) if type(edges) == list else edges
    uh, uIdxs = np.unique(cantorPiV(edges), return_index=True)
    return edges[uIdxs]


def computePolygonCentroid2D(pts, withArea=False):
    rPts = np.roll(pts, 1, axis=0)
    w = pts[:, 0] * rPts[:, 1] - rPts[:, 0] * pts[:, 1]
    area = w.sum() / 2.0
    centroid = np.sum((pts + rPts) * w.reshape(-1, 1), axis=0) / (6 * area)
    return (centroid, np.abs(area)) if withArea else centroid


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


def writeObjFile(filePath, vertices, faces, edges=[], comment="", subTags=[]):
    fh = open(filePath, "w")
    fh.write("# obj export\n")
    if len(comment):
        fh.write(comment)
    if not len(subTags):
        fh.write("o %s\n" % os.path.basename(filePath)[:-4])
        for vertex in vertices:
            fh.write("v %0.6f %0.6f %0.6f\n" % tuple(vertex))
        for face in faces:
            fh.write(("f" + " %d" * len(face) + "\n") % tuple(face + 1))
        for edge in edges:
            fh.write(("l" + " %d" * len(edge) + "\n") % tuple(edge + 1))
    else:
        vOffset = 0
        for subIdx, tag in enumerate(subTags):
            fh.write("o %s\n" % tag)
            for vertex in vertices[subIdx]:
                fh.write("v %0.6f %0.6f %0.6f\n" % tuple(vertex))
            for face in faces[subIdx]:
                fh.write(("f" + " %d" * len(face) + "\n") % tuple(face + vOffset + 1))
            if len(edges):
                for edge in edges[subIdx]:
                    fh.write(("l" + " %d" * len(edge) + "\n") % tuple(edge + vOffset + 1))
            vOffset += len(vertices[subIdx])
    fh.close()
