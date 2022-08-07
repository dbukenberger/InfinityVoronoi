from util import *

#   2 - n - 3
#   | \   / |
#   w   0   e
#   | /   \ |
#   1 - s - 4

boxVerts = np.float32([[0, 0], [-1, -1], [-1, 1], [1, 1], [1, -1]])
edgeCenters = np.vstack([np.eye(2), -np.eye(2), boxVerts[1:] * 0.5])
edgeNormals = np.vstack([np.eye(2), -np.eye(2), boxVerts[1:]])

#           e:   +x    n:   +y    w:   -x    s:   -y
triIdxs = [[0, 3, 4], [0, 2, 3], [0, 1, 2], [0, 4, 1]]
ecIdxs =  [[6, 0, 7], [5, 1, 6], [4, 2, 5], [7, 3, 4]]
enIdxs =  [[5, 0, 4], [4, 1, 7], [7, 2, 6], [6, 3, 5]]


class TriCutObject:
    def __init__(self, site, di, scale, M):

        self.edges = list(map(np.int32, [[0, 1], [1, 2], [2, 0]]))
        self.polys = {1: [0, 1, 2]}
        self.edgePolyIdxs = [np.int64([1, -1]) for e in self.edges]

        vertScales = [[1, 1], [scale[(di + 1) % 4], scale[di]], [scale[di - 1], scale[di]]] if di % 2 else [[1, 1], [scale[di], scale[di + 1]], [scale[di], scale[(di - 1) % 4]]]
        eCenterScales = [[scale[(di + 1) % 4], scale[di]], [1, scale[di]], [scale[(di - 1) % 4], scale[di]]] if di % 2 else [[scale[di], scale[(di + 1) % 4]], [scale[di], 1], [scale[di], scale[(di - 1) % 4]]]
        eNormalScales = [[scale[di], scale[(di + 1) % 4]], [1, 1], [scale[di], scale[(di - 1) % 4]]] if di % 2 else [[scale[(di + 1) % 4], scale[di]], [1, 1], [scale[(di - 1) % 4], scale[di]]]       

        self.vertices = site + np.dot(boxVerts[triIdxs[di]] * vertScales, M.T)
        eCenters = site + np.dot(edgeCenters[ecIdxs[di]] * eCenterScales, M.T)
        eNormals = normVec(np.dot(edgeNormals[enIdxs[di]] * eNormalScales, M.T))

        self.edgesPlanes = {-(i + 4): [eCenters[i], eNormals[i]] for i in range(3)}
        self.edgePlaneKeys = [-4, -5, -6]

    def cutWithLine(self, o, n, cutPlaneKey):
        dots = np.dot(self.vertices - o, n)
        inside = dots <= eps
        onLine = np.abs(dots) < eps

        vMasks = simpleSign(dots, eps)
        if np.all(vMasks > 0) or np.all(vMasks < 0):
            return

        edgeMasks = [vMasks[edge] for edge in self.edges]
        edgeHashs = cantorPiV(np.int32(self.edges))

        newPolys = {}
        cutPolyKeys = set()
        for polyKey in self.polys.keys():

            signs = set()
            for eIdx in self.polys[polyKey]:
                signs.update(edgeMasks[eIdx])

            if 1 in signs and -1 in signs:
                cutPolyKeys.add(polyKey)
                newPolys[polyKey * 2] = []
                newPolys[polyKey * 2 + 1] = []
            else:
                newPolys[polyKey] = self.polys[polyKey]

        cutEdgesMasks = {}
        oldEdgesMasks = {}
        newEdgePolyIdxs = []
        for edgeMask, edgeHash, epi in zip(edgeMasks, edgeHashs, self.edgePolyIdxs):
            if all(edgeMask <= 0):
                oldEdgesMasks[edgeHash] = edgeMask
                if epi[0] in cutPolyKeys and epi[1] in cutPolyKeys:
                    newEdgePolyIdxs.append(epi * 2)
                elif epi[0] in cutPolyKeys:
                    newEdgePolyIdxs.append(epi * [2, 1])
                elif epi[1] in cutPolyKeys:
                    newEdgePolyIdxs.append(epi * [1, 2])
                else:
                    newEdgePolyIdxs.append(epi)
            elif all(edgeMask >= 0):
                oldEdgesMasks[edgeHash] = edgeMask
                if epi[0] in cutPolyKeys and epi[1] in cutPolyKeys:
                    newEdgePolyIdxs.append(epi * 2 + 1)
                elif epi[0] in cutPolyKeys:
                    newEdgePolyIdxs.append(epi * [2, 1] + [1, 0])
                elif epi[1] in cutPolyKeys:
                    newEdgePolyIdxs.append(epi * [1, 2] + [0, 1])
                else:
                    newEdgePolyIdxs.append(epi)
            else:
                newEdgePolyIdxs.append(epi)
                cutEdgesMasks[edgeHash] = edgeMask

        numVerts = len(self.vertices)
        cutPlaneKeys = []
        edgesReplaced = {}
        edgeUpdates = []
        for cutPolyKey in cutPolyKeys:

            newEdgeInner = []

            eIdxs = self.polys[cutPolyKey]
            for eIdx in eIdxs:
                edge = self.edges[eIdx]

                if vMasks[edge[0]] == 0 and edge[0] not in newEdgeInner:
                    newEdgeInner.append(edge[0])
                if vMasks[edge[1]] == 0 and edge[1] not in newEdgeInner:
                    newEdgeInner.append(edge[1])

                edgeHash = edgeHashs[eIdx]
                if edgeHash in cutEdgesMasks.keys():
                    cutEdgeMask = cutEdgesMasks[edgeHash]

                    if edgeHash in edgesReplaced.keys():
                        newVertIdx, eJdx = edgesReplaced[edgeHash]
                    else:
                        newVertIdx = numVerts
                        numVerts += 1

                        cutPlaneKeys.append(self.edgePlaneKeys[eIdx])

                        eJdx = len(self.edges)
                        edgesReplaced[edgeHash] = [newVertIdx, eJdx]

                        # self.edges[eIdx][1] = newVertIdx # first half
                        edgeUpdates.append([eIdx, newVertIdx])  # update later
                        self.edges.append(np.int32([newVertIdx, edge[1]]))  # second half

                        # self.edgePlaneKeys[eIdx] # first half unchanged
                        self.edgePlaneKeys.append(self.edgePlaneKeys[eIdx])  # second half

                        if cutEdgeMask[0] > 0 and cutEdgeMask[1] < 0:  # 1 -> 0
                            newEdgePolyIdxs.append(newEdgePolyIdxs[eIdx] * 2)
                            newEdgePolyIdxs[eIdx] *= 2
                            newEdgePolyIdxs[eIdx] += 1

                        if cutEdgeMask[1] > 0 and cutEdgeMask[0] < 0:  # 0 -> 1
                            newEdgePolyIdxs.append(newEdgePolyIdxs[eIdx] * 2 + 1)
                            newEdgePolyIdxs[eIdx] *= 2

                    newEdgeInner.append(newVertIdx)  # new inner

                    for nepi in newEdgePolyIdxs[eIdx]:
                        if nepi > 0 and eIdx not in newPolys[nepi]:
                            newPolys[nepi].append(eIdx)
                    for nepi in newEdgePolyIdxs[eJdx]:
                        if nepi > 0 and eJdx not in newPolys[nepi]:
                            newPolys[nepi].append(eJdx)

                else:
                    edgeMask = oldEdgesMasks[edgeHash]
                    if np.all(edgeMask <= 0):
                        newPolys[cutPolyKey * 2].append(eIdx)
                    elif np.all(edgeMask >= 0):
                        newPolys[cutPolyKey * 2 + 1].append(eIdx)

            assert len(newEdgeInner) == 2, "oh oh, this should not happen"

            newPolys[cutPolyKey * 2].append(len(self.edges))
            newPolys[cutPolyKey * 2 + 1].append(len(self.edges))
            newEdgePolyIdxs.append(np.int64([cutPolyKey * 2, cutPolyKey * 2 + 1]))

            self.edges.append(np.int32(newEdgeInner))
            self.edgePlaneKeys.append(cutPlaneKey)
        self.edgesPlanes[cutPlaneKey] = [o, n]

        for eIdx, vIdx in edgeUpdates:
            self.edges[eIdx][1] = vIdx

        if len(cutPlaneKeys):
            ePs = [self.edgesPlanes[cpKey] for cpKey in cutPlaneKeys]
            newVerts = intersectLinesLine2D(np.float32(ePs), o, n)

            self.vertices = np.vstack([self.vertices, newVerts])

        self.polys = newPolys
        self.edgePolyIdxs = newEdgePolyIdxs

    def setPolyIoLabels(self, msk):
        if not hasattr(self, "polysIoLabels"):
            self.polysIoLabels = {pk: True for pk in self.polys.keys()}
            self.edgePolyIdxs = np.int64(self.edgePolyIdxs)

        self.cellPolyIdxs = []
        for pIdx, (pk, io) in enumerate(zip(self.polys.keys(), msk)):
            self.polysIoLabels[pk] = io
            if io:
                self.cellPolyIdxs.append(pIdx)
            else:
                self.edgePolyIdxs[self.edgePolyIdxs == pk] *= 0

    def getHullVerts(self):
        es = {}
        for e, ePolyIdx, ePlaneKey in zip(self.edges, self.edgePolyIdxs, self.edgePlaneKeys):
            if simpleSign(ePolyIdx).sum() == 1:
                if ePlaneKey in es.keys():
                    es[ePlaneKey].append(e)
                else:
                    es[ePlaneKey] = [e]

        segs = []
        for epk in es.keys():
            ces = findConnectedEdgeSegments(es[epk])
            for ce in ces:
                ep = edgesToPath(ce)
                segs.append([ep[0], ep[-1]])

        self.hullPlaneKeys = list(es.keys())
        return [self.vertices[seg] for seg in edgesToPaths(segs)]

    def computeCentroids(self):
        self.polysCentroids = np.empty((len(self.polys), 2), np.float32)
        self.polysAreas = np.empty(len(self.polys), np.float32)
        for pIdx, pk in enumerate(self.polys.keys()):
            es = [self.edges[eIdx].tolist() for eIdx in self.polys[pk]]
            self.polysCentroids[pIdx], self.polysAreas[pIdx] = computePolygonCentroid2D(self.vertices[edgesToPath(es)], True)

        return self.polysCentroids

    def plot(self):
        if mplMissing:
            print('matplotlib missing')
            return
        
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        for pKey in self.polys.keys():
            face = edgesToPath([self.edges[eIdx].tolist() for eIdx in self.polys[pKey]])
            cVerts = self.vertices[face]
            cVerts = cVerts - (cVerts - cVerts.mean(axis=0)) * 0.05
            ax.fill(cVerts[:, 0], cVerts[:, 1], fill=self.polysIoLabels[pKey] if hasattr(self, "polysIoLabels") else False)

        for vIdx, vt in enumerate(self.vertices):
            ax.text(vt[0], vt[1], str(vIdx))

        ax.set_aspect("equal", "box")
        plt.show()


# principle bisector plane computation
def computeCutPlaneMP(cpChunk):
    A, B, a, b, lambdaA, lambdaB = cpChunk
    adb = np.dot(a, b)

    if abs(1 - adb) < eps:  # same dir
        if np.abs(np.dot(a, A - B)) < eps:  # same pos
            n = normVec(A - B)
        elif lambdaA == lambdaB:  # same weight -> no plane
            n = a * 0
        else:
            n = a
        o = (A * lambdaB + B * lambdaA) / (lambdaA + lambdaB)
    else:
        n = normVec(a / lambdaA - b / lambdaB)

        o = A + a * np.dot(A - B, b * lambdaA / lambdaB) / (1 - np.dot(a, b * lambdaA / lambdaB))
        #o = B + b * np.dot(B - A, a * lambdaB / lambdaA) / (1 - np.dot(a * lambdaB / lambdaA, b))
        # should be equivalent


    return [o, n]


# same as above but vectorized for multiple sites
def computeCutPlanes(sitesA, sitesB, vecsA, vecsB, lambdasA, lambdasB):
    adbs = inner1d(vecsA, vecsB)
    BtoA = sitesA - sitesB

    bDiv = vecsB * (lambdasA / lambdasB)
    pOs = sitesA + vecsA * (inner1d(BtoA, bDiv) / (1 - inner1d(vecsA, bDiv))).reshape(-1, 1)
    pNs = normVec(vecsA / lambdasA - vecsB / lambdasB)

    sameDirMask = np.abs(1 - adbs) < eps
    samePosMask = np.bitwise_and(sameDirMask, np.abs(inner1d(vecsA, BtoA)) < eps)
    lambdasMask = np.bitwise_and(sameDirMask, (lambdasA == lambdasB)[:, 0])

    pNs[sameDirMask] = vecsA[sameDirMask]
    pNs[lambdasMask] *= 0
    pNs[samePosMask] = normVec(BtoA[samePosMask])
    pOs[samePosMask] = (sitesA[samePosMask] * lambdasB[samePosMask] + sitesB[samePosMask] * lambdasA[samePosMask]) / (lambdasA[samePosMask] + lambdasB[samePosMask])

    return np.transpose(np.dstack([pOs, pNs]), axes=[0, 2, 1])


def extractCellVertsMP(cellTris):
    # geometry of the individual sectors
    cVertsENWS = [cellTri.getHullVerts() for cellTri in cellTris]

    # a cell may consist of separate parts
    cVerts = []
    for cVertsP in cVertsENWS:
        for cvn in cVertsP:
            for i, cv in enumerate(cVerts):
                if norm(cvn[-1] - cv[0]) < eps:
                    cVerts[i] = np.vstack([cvn, cv])
                    break
                if norm(cvn[0] - cv[0]) < eps:
                    cVerts[i] = np.vstack([cvn[::-1], cv])
                    break
                if norm(cvn[0] - cv[-1]) < eps:
                    cVerts[i] = np.vstack([cv, cvn])
                    break
                if norm(cvn[-1] - cv[-1]) < eps:
                    cVerts[i] = np.vstack([cv, cvn[::-1]])
                    break
            else:
                cVerts.append(cvn)

    cVertsClean = [limitedDissolve2D(cvs) for cvs in cVerts]
    return cVertsClean, flatten([cellTri.hullPlaneKeys for cellTri in cellTris])


def cutWithPlanesMP(qTuple):
    cellTri, cpPacks = qTuple
    for (o, n), cutPlaneKey in cpPacks:
        cellTri.cutWithLine(o, n, cutPlaneKey)
    cellTri.computeCentroids()
    return cellTri


class ChebyshevObject2D:

    domainPlanes = np.repeat(np.vstack([np.eye(2), -np.eye(2)]), 2, axis=0).reshape(4, 2, 2)

    def __init__(self, sites, oriFun=None, aniFun=None, extent=1, withMP=True, tag = ''):
        self.sites = sites
        self.numSites = len(self.sites)
        self.colors = np.uint8([hex2rgb(seed2hex(i)) for i in range(self.numSites)])
        self.sitesNeighbors = [[[] for i in range(4)] for sIdx in range(self.numSites)]

        self.oriFun = oriFun
        self.aniFun = aniFun

        self.domainExtent = extent
        self.domainPlanes *= [[[self.domainExtent], [1]]]
        self.mpPool = mp.Pool(processes=cpuCount) if withMP else None

        self.cellScale = 2 * self.domainExtent / np.sqrt(self.numSites)
        self.cellScales = np.ones(self.numSites) * self.cellScale

        self.log = Logger(tag if tag else 'infinityVoronoi')
        self.log.logThis('Sites', self.numSites)
        self.log.logThis('CPUs', cpuCount if withMP else 1)

        self.timings = []
        self.initCells()

    def initCells(self):
        st = time()

        if self.oriFun is None:
            self.Ms = np.float32([np.eye(2)] * self.numSites)
        else:
            self.Ms = np.float32([Mr2D(angle) for angle in self.oriFun(self.sites)])
        self.Mvecs = np.float32([np.vstack([M.T, -M.T]) for M in self.Ms])

        if self.aniFun is None:
            self.lambdas = np.ones((self.numSites, 4), np.float32)
        else:
            self.lambdas = self.aniFun(self.sites)

        # geometry (o, n) and topology (sIdx, sJdx)
        self.cutPlanes = {-k: [self.domainPlanes[k], []] for k in range(4)}
        self.cutPlaneKeys = [[[] for j in range(4)] for i in range(self.numSites)]

        self.cellTris = [[] for i in range(self.numSites)]
        for sIdx in range(self.numSites):
            for di in range(4):
                self.cellTris[sIdx].append(TriCutObject(self.sites[sIdx], di, self.cellScales[sIdx] * self.lambdas[sIdx], self.Ms[sIdx]))
                vs = self.cellTris[sIdx][-1].vertices

                cpMins = vs.min(axis=0)
                cpMaxs = vs.max(axis=0)
                for d in range(2):
                    if cpMaxs[d] > self.domainExtent:
                        self.cutPlaneKeys[sIdx][di].append(-d)
                    if cpMins[d] < -self.domainExtent:
                        self.cutPlaneKeys[sIdx][di].append(-(d + 2))

        self.timings.append([time() - st])

    def logStats(self, latestOnly=True):
        tms = np.float32(self.timings[-1]) if latestOnly else np.sum(self.timings, axis=0)
        self.log.logThis("Stats", ["Init", "Ps", "Cut", "Clip", "Diss", "Total"], ",".join(["% 7s"] * 6))
        self.log.logThis("in %", (100 * tms / tms.sum()).tolist() + [tms.sum()], ",".join(["%7.2f"] * 6) + "s")

    def finish(self):
        self.logStats(False)
        if self.mpPool is not None:
            self.mpPool.close()
            self.mpPool.join()

        self.log.logThis("#/Cell", ["Min", "Max", "Med"], ",".join(["% 4s"] * 3))
        numNeighbors = list(map(len, self.cellAdjacency))
        numParts = list(map(len, self.cellVertexSets))
        numVerts = list(map(len, map(np.concatenate, self.cellVertexSets)))
        for nVals, tag in zip([numNeighbors, numParts, numVerts], ["Adj.", "Parts", "Verts"]):
            self.log.logThis(tag, [min(nVals), max(nVals), np.median(nVals)], ",".join(["% 4d"] * 3))

    def computeNeighborsAndPlanes(self):
        siteDimTuples = []
        siteTrisVerts = []
        for sIdx in tqdm(range(self.numSites), ascii=True, desc="knnPlanes" if self.mpPool is None else "serializing"):
            for sJdx in range(sIdx + 1, self.numSites):
                if norm(self.sites[sIdx] - self.sites[sJdx]) > self.cellScale * 4:
                    continue
                for di, cTri in enumerate(self.cellTris[sIdx]):
                    for dj, cTrj in enumerate(self.cellTris[sJdx]):
                        if self.mpPool is None and trianglesDoIntersect2D(cTri.vertices, cTrj.vertices):
                            # without MP, compute and add planes on the spot and then return
                            o, n = computeCutPlaneMP((self.sites[sIdx], self.sites[sJdx], self.Mvecs[sIdx, di], self.Mvecs[sJdx, dj], self.lambdas[sIdx, di], self.lambdas[sJdx, dj]))
                            self._addCutPlane(sIdx, di, sJdx, dj, o, n)
                        else:
                            # otherwise serialize first and compute everything in parallel afterwards
                            siteDimTuples.append([sIdx, di, sJdx, dj])
                            siteTrisVerts.append([cTri.vertices, cTrj.vertices])

        if self.mpPool is None:
            return

        overlapMask = list(tqdm(self.mpPool.imap(trianglesDoIntersect2D, siteTrisVerts, chunksize=cpuCount * 4), total=len(siteTrisVerts), ascii=True, desc="kNN@%d" % cpuCount))
        siteDimTuples = np.int32(siteDimTuples)[overlapMask]

        # vectorized version faster than multiprocessing due to overhead
        ons = computeCutPlanes(
            self.sites[siteDimTuples[:, 0]],
            self.sites[siteDimTuples[:, 2]],
            self.Mvecs[siteDimTuples[:, 0], siteDimTuples[:, 1]],
            self.Mvecs[siteDimTuples[:, 2], siteDimTuples[:, 3]],
            self.lambdas[siteDimTuples[:, 0], siteDimTuples[:, 1]].reshape(-1, 1),
            self.lambdas[siteDimTuples[:, 2], siteDimTuples[:, 3]].reshape(-1, 1))

        # multiprocessing version
        # cpChunks = [[self.sites[sIdx], self.sites[sJdx], self.Mvecs[sIdx, di], self.Mvecs[sJdx, dj], self.lambdas[sIdx, di], self.lambdas[sJdx, dj]] for sIdx, di, sJdx, dj in siteDimTuples]
        # ons = list(tqdm(self.mpPool.imap(computeCutPlaneMP, cpChunks, chunksize=cpuCount*4), total=len(cpChunks), ascii=True, desc="planes@%d"%cpuCount))

        for (sIdx, di, sJdx, dj), (o, n) in tqdm(zip(siteDimTuples, ons), total=len(siteDimTuples), ascii=True, desc="addPlanes"):
            self._addCutPlane(sIdx, di, sJdx, dj, o, n)

    def _addCutPlane(self, sIdx, di, sJdx, dj, o, n):
        self.sitesNeighbors[sIdx][di].append((sJdx, dj))
        self.sitesNeighbors[sJdx][dj].append((sIdx, di))

        if norm(n) < eps:
            return

        cutPlaneKey = cantorPi(cantorPiO(sIdx, di), cantorPiO(sJdx, dj))
        self.cutPlanes[cutPlaneKey] = [[o, n], [sIdx, sJdx]]

        addIt = False
        for cpKey in self.cutPlaneKeys[sIdx][di]:
            on = self.cutPlanes[cpKey][0]
            if np.abs(np.dot(o - on[0], on[1])) < eps and vecsParallel(n, on[1]):
                break
        else:
            self.cutPlaneKeys[sIdx][di].append(cutPlaneKey)

        for cpKey in self.cutPlaneKeys[sJdx][dj]:
            on = self.cutPlanes[cpKey][0]
            if np.abs(np.dot(o - on[0], on[1])) < eps and vecsParallel(n, on[1]):
                break
        else:
            self.cutPlaneKeys[sJdx][dj].append(cutPlaneKey)

    def cutWithPlanes(self):
        """
        # basic functionality without MP
        for sIdx in tqdm(range(self.numSites), ascii=True, desc='cutting'):
            for di in range(4):
                for cutPlaneKey in self.cutPlaneKeys[sIdx][di]:
                    o,n = self.cutPlanes[cutPlaneKey][0]
                    self.cellTris[sIdx][di].cutWithLine(o, n, cutPlaneKey)
                self.cellTris[sIdx][di].computeCentroids()
        """

        cpPacks = []
        for sIdx in range(self.numSites):
            for di in range(4):
                cpPack = [self.cellTris[sIdx][di], []]
                for cutPlaneKey in self.cutPlaneKeys[sIdx][di]:
                    cpPack[1].append([self.cutPlanes[cutPlaneKey][0], cutPlaneKey])
                cpPacks.append(cpPack)

        if self.mpPool is None:
            res = list(tqdm(map(cutWithPlanesMP, cpPacks), total=len(cpPacks), ascii=True, desc="cutting"))
        else:
            res = list(tqdm(self.mpPool.imap(cutWithPlanesMP, cpPacks), total=len(cpPacks), ascii=True, desc="cutting@%d" % cpuCount))

        for sIdx in range(self.numSites):
            for di in range(4):
                self.cellTris[sIdx][di] = res.pop(0)

    def clipOuterGeometry(self):
        for sIdx in tqdm(range(self.numSites), ascii=True, desc="clipping"):
            for di in range(4):
                polysCentroids = self.cellTris[sIdx][di].polysCentroids
                iDists = np.dot(polysCentroids - self.sites[sIdx], self.Mvecs[sIdx, di]) / self.lambdas[sIdx, di]

                sJdxs = np.unique([sJdx for sJdx, dj in self.sitesNeighbors[sIdx][di]])

                msk = np.abs(polysCentroids).max(axis=1) < self.domainExtent
                for sJdx in sJdxs:
                    jDists = np.dot(polysCentroids - self.sites[sJdx], self.Mvecs[sJdx].T) / self.lambdas[sJdx]
                    msk *= iDists < jDists.max(axis=1)

                # vectorized but not faster
                # jDists = innerNxM(polysCentroids - self.sites[sJdxs].reshape((-1,1,2)), self.Mvecs[sJdxs]) / self.lambdas[sJdxs].reshape(-1,1,4)
                # msk = np.bitwise_and(np.abs(polysCentroids).max(axis=1) < self.domainExtent, iDists < jDists.max(axis=2).min(axis=0))

                self.cellTris[sIdx][di].setPolyIoLabels(msk)

    def dissolveCells(self):
        if self.mpPool is None:
            res = list(tqdm(map(extractCellVertsMP, self.cellTris), total=self.numSites, ascii=True, desc="cellVerts"))
        else:
            res = list(tqdm(self.mpPool.imap(extractCellVertsMP, self.cellTris), total=self.numSites, ascii=True, desc="cellVerts@%d" % cpuCount))
        self.cellVertexSets, cellPlaneKeys = list(zip(*res))

        self.cellCentroids = np.empty((self.numSites, 2), np.float32)
        self.cellBBs = np.empty((self.numSites, 4, 2), np.float32)
        self.cellBBcenters = np.empty((self.numSites, 2), np.float32)
        self.cellAdjacency = []
        for sIdx in tqdm(range(self.numSites), total=self.numSites, ascii=True, desc="dissolving"):

            # L2 centroids
            triPolyCentroids = np.vstack([cellTri.polysCentroids[cellTri.cellPolyIdxs] for cellTri in self.cellTris[sIdx]])
            triPolyAreas = np.concatenate([cellTri.polysAreas[cellTri.cellPolyIdxs] for cellTri in self.cellTris[sIdx]])
            self.cellCentroids[sIdx] = np.dot(triPolyAreas, triPolyCentroids) / np.sum(triPolyAreas)

            # bounding boxes and BB centers
            cellVerts = np.dot(np.vstack(self.cellVertexSets[sIdx]), self.Ms[sIdx])
            minMax = np.float32([cellVerts.min(axis=0), cellVerts.max(axis=0)])
            bbCenter = minMax.mean(axis=0)
            self.cellBBcenters[sIdx] = np.dot(bbCenter, self.Ms[sIdx].T)
            self.cellBBs[sIdx] = np.dot(boxVerts[1:] * np.abs(minMax[1] - minMax[0]) / 2 + bbCenter, self.Ms[sIdx].T)

            # site indices of adjacent cells
            adjacentSites = flatten([self.cutPlanes[planeKey][1] for planeKey in cellPlaneKeys[sIdx] if planeKey in self.cutPlanes.keys()])
            self.cellAdjacency.append(np.unique([sJdx for sJdx in adjacentSites if sJdx != sIdx]))

        self.cellAdjacencyEdges = filterForUniqueEdges(flatten([[[sIdx, aIdx] for aIdx in self.cellAdjacency[sIdx]] for sIdx in range(self.numSites)]))

    def computeDiagram(self, finish=True):
        for fun in [self.computeNeighborsAndPlanes, self.cutWithPlanes, self.clipOuterGeometry, self.dissolveCells]:
            st = time()
            fun()
            self.timings[-1].append(time() - st)

        if finish:
            self.finish()
        else:
            self.logStats()

    def lloydRelax(self, itersThresh=0, export=False):
        if not hasattr(self, "cellCentroids"):
            self.computeDiagram(False)

        iters = 0
        while True:
            if export:
                self.exportToObj("lloyd%04d.obj" % iters)

            moveDists = norm(self.sites - self.cellCentroids)
            self.log.logThis("Iter", (iters, int(itersThresh), moveDists.max(), (type(itersThresh) == float) * itersThresh), "% 4d / % 4d, MD: %0.6f > %0.6f")
            if (itersThresh < 1 and moveDists.max() < itersThresh) or iters == itersThresh:
                break

            # use L2 centroids not BB centers
            self.sites = self.cellCentroids.copy()
            self.initCells()
            self.computeDiagram(False)
            iters += 1

        self.finish()

    def plot(self, withSites=True, withCentroids=True, withAdjacency=True, withBBs=True, fileName=""):
        if mplMissing:
            print('matplotlib missing')
            return
        
        fig = plt.figure("Voronoi Plot")
        ax = fig.add_axes([0, 0, 1, 1])

        legendElements = []

        for sIdx in range(self.numSites):
            for cellVerts in self.cellVertexSets[sIdx]:
                ax.fill(cellVerts[:, 0], cellVerts[:, 1], color=self.colors[sIdx] / 255)

        if withAdjacency:
            adjacencyPlot = ax.plot(self.sites[self.cellAdjacencyEdges, 0].T, self.sites[self.cellAdjacencyEdges, 1].T, color="white", linewidth=1, label="Adjacency")
            legendElements.append(adjacencyPlot[0])

        if withSites:
            dirVecs = np.tile([edgeNormals[:4]], [self.numSites, 1, 1]) * self.lambdas.reshape(-1, 4, 1)
            oriPts = self.sites.reshape(-1, 1, 2) + innerNxM(dirVecs, self.Ms) * self.cellScale / 5
            sitesPlot = ax.plot(np.vstack(oriPts[:, [[0, 2], [1, 3]], 0]).T, np.vstack(oriPts[:, [[0, 2], [1, 3]], 1]).T, color="black")
            legendElements.append(ax.scatter([], [], marker="+", color="black", label="Sites"))

        if withCentroids:
            centroidsPlot = ax.scatter(self.cellCentroids[:, 0], self.cellCentroids[:, 1], color="yellow", s=15, label="$L_2$ Centroids")
            legendElements.append(centroidsPlot)

        if withBBs:
            bbCentersPlot = ax.scatter(self.cellBBcenters[:, 0], self.cellBBcenters[:, 1], color="green", s=15, label="$L_\infty$ BB Centers")
            legendElements.append(bbCentersPlot)

            for sIdx in range(self.numSites):
                ax.plot(self.cellBBs[sIdx][[0, 1, 2, 3, 0], 0], self.cellBBs[sIdx][[0, 1, 2, 3, 0], 1], color=self.colors[sIdx] / 255, linewidth=1)

        ax.set_aspect("equal", "box")
        fig.legend(handles=legendElements, facecolor="gray")
        plt.axis(False)
        if fileName:
            plt.savefig(resDir + fileName)
        plt.show()

    def exportToObj(self, fileName="infinityVoronoi.obj", withSites=True, withCentroids=True, withAdjacency=True, withBBs=True):
        numVerts = 0
        verts = []
        cells = []
        for sIdx in range(self.numSites):
            for cellVerts in self.cellVertexSets[sIdx]:
                cells.append(np.arange(len(cellVerts)) + numVerts)
                verts.append(cellVerts)
                numVerts += len(cellVerts)

        vertices = [pad2Dto3D(np.vstack(verts))]
        faces = [cells]
        edges = [[]]
        tags = ["cells"]

        if withSites:
            vertices.append(pad2Dto3D(self.sites))
            faces.append([])
            edges.append([])
            tags.append("sites")

        if withCentroids:
            vertices.append(pad2Dto3D(self.cellCentroids))
            faces.append([])
            edges.append([])
            tags.append("centroids")

        if withAdjacency:
            vertices.append(pad2Dto3D(self.sites))
            faces.append([])
            edges.append(self.cellAdjacencyEdges)
            tags.append("edges")

        if withBBs:
            bbVerts = []
            bbEdges = []
            for sIdx in range(self.numSites):
                bbVerts.append(np.vstack([self.cellBBs[sIdx], self.cellBBcenters[sIdx]]))
                bbEdges.append(np.int32([[0, 1], [1, 2], [2, 3], [3, 0]]) + sIdx * 5)

            vertices.append(pad2Dto3D(np.vstack(bbVerts)))
            faces.append([])
            edges.append(np.vstack(bbEdges))
            tags.append("BBs")

        writeObjFile(resDir + fileName, vertices, faces, edges, subTags=tags)

    def exportToSvg(self, fileName="infinityVoronoi.svg", size=512, withSites = True):
        theDom = minidom.parseString('<svg xmlns="http://www.w3.org/2000/svg" height="%d" width="%d"></svg>'%(size, size))
        for sIdx in range(self.numSites):
            g = theDom.createElement('g')
            g.setAttribute('style', 'fill:%s;'%rgb2hex(self.colors[sIdx]))
            theDom.childNodes[0].appendChild(g)
            for cellVerts in self.cellVertexSets[sIdx]:
                p = theDom.createElement('path')
                verts = (cellVerts * [1,-1] + 1) / 2 * size
                p.setAttribute('d', 'M' + (('L%.6f %.6f ' * len(verts))%tuple(verts.ravel()))[1:] + 'Z')
                g.appendChild(p)

        if withSites:
            g = theDom.createElement('g')
            g.setAttribute('style', 'stroke:#000;stroke-width:2px;stroke-linecap:round;')
            theDom.childNodes[0].appendChild(g)
            for sIdx in range(self.numSites):                
                aniVecs = edgeNormals[:4] * self.lambdas[sIdx].reshape(-1, 1)
                aniOriVecs = self.sites[sIdx] + np.dot(aniVecs, self.Ms[sIdx].T) * self.cellScale / 5
                aniOriPts = (aniOriVecs * [1, -1] + 1) / 2 * size
            
                p = theDom.createElement('path')
                p.setAttribute('d', 'M%.6f %.6f L%.6f %.6f'%(aniOriPts[0, 0], aniOriPts[0, 1], aniOriPts[2, 0], aniOriPts[2, 1]))
                g.appendChild(p)

                p = theDom.createElement('path')
                p.setAttribute('d', 'M%.6f %.6f L%.6f %.6f'%(aniOriPts[1, 0], aniOriPts[1, 1], aniOriPts[3, 0], aniOriPts[3, 1]))
                g.appendChild(p)

        fh = open(resDir + fileName, 'w')
        fh.write(theDom.toprettyxml())
        fh.close()


if __name__ == "__main__":

    n = 20
    e = 1.0

    rng = np.linspace(-e, e, n, endpoint=False) + e / n
    xyGrid = np.vstack(np.dstack(np.meshgrid(rng, rng)))
    sites = xyGrid + randomJitter(n**2, 2, e / n)


    def oriFun(xys):
        xs,ys = xys.T
        uvs = (xys + [0.5,0]) / np.sqrt((xs + 0.5)**2 + ys**2).reshape(-1,1)**3 + (xys - [0.5,0]) / np.sqrt((xs - 0.5)**2 + ys**2).reshape(-1,1)**3
        a = np.arctan2(uvs[:,0], uvs[:,1])
        m = np.abs(xs) < 0.5
        a[m] = np.cos(xs[m] * np.pi) * -np.sign(ys[m]) * np.sign(xs[m])
        return np.pi - a


    def aniFun(xys):
        xs,ys = xys.T
        s = np.min(np.abs(ys + [[0.5], [-0.5]] * np.sin(xs*np.pi)), axis=0)
        s[xs>0.5] = np.abs(0.5 - norm([0.5,0] - xys[xs>0.5]))
        s[xs<-0.5] = np.abs(0.5 - norm([-0.5,0] - xys[xs<-0.5]))
        return np.tile([2-s, np.ones(len(xys))], [2,1]).T


    cObj = ChebyshevObject2D(sites, oriFun, aniFun, e, withMP=True)
    cObj.lloydRelax(0.001)
    cObj.exportToSvg()
