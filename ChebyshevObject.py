from util import *
from TriCutObject import *
from PyraCutObject import *
from DissCell import *


def sectorsDoIntersectMP(sGeometry):
    if len(sGeometry) == 2:
        return trianglesDoIntersect2D(sGeometry[0], sGeometry[1])
    elif len(sGeometry) == 4:
        return pyrasDoIntersect(sGeometry[0], sGeometry[1], sGeometry[2], sGeometry[3])


# principle bisector plane computation
def computeCutPlaneMP(cpPack):
    A, B, a, b, lambdaA, lambdaB = cpPack
    adb = np.dot(a, b)

    o = None
    if abs(1 - adb) < eps:  # a and b parallel
        if np.abs(np.dot(a, A - B)) < eps:  # a and b orthogonal to AB
            o = (A * lambdaB + B * lambdaA) / (lambdaA + lambdaB)
            n = normVec(A - B)
        elif lambdaA == lambdaB:  # same weight -> no plane
            o = a * 0
            n = a * 0
        else:
            n = a
    else:
        n = normVec(a / lambdaA - b / lambdaB)

    if o is None:
        o = A + a * np.dot(A - B, b * lambdaA / lambdaB) / (1 - np.dot(a, b * lambdaA / lambdaB))
        #o = B + b * np.dot(B - A, a * lambdaB / lambdaA) / (1 - np.dot(a * lambdaB / lambdaA, b))
        # should be equivalent

    return o, n


# same as above but vectorized for multiple sites
def computeCutPlanes(sitesA, sitesB, vecsA, vecsB, lambdasA, lambdasB):
    adbs = inner1d(vecsA, vecsB)
    BtoA = sitesA - sitesB

    sameDirMask = np.abs(1 - adbs) < eps
    samePosMask = np.bitwise_and(sameDirMask, np.abs(inner1d(vecsA, BtoA)) < eps)
    lambdasMask = np.bitwise_and(sameDirMask, (lambdasA == lambdasB)[:, 0])

    bDiv = vecsB * (lambdasA / lambdasB)
    bDiv[lambdasMask] *= -1 # to avoid division by 0
    pOs = sitesA + vecsA * (inner1d(BtoA, bDiv) / (1 - inner1d(vecsA, bDiv))).reshape(-1, 1)
    pNs = normVec(vecsA / lambdasA - vecsB / lambdasB)

    pNs[sameDirMask] = vecsA[sameDirMask]
    pNs[lambdasMask] *= 0
    pNs[samePosMask] = normVec(BtoA[samePosMask])
    pOs[samePosMask] = (sitesA[samePosMask] * lambdasB[samePosMask] + sitesB[samePosMask] * lambdasA[samePosMask]) / (lambdasA[samePosMask] + lambdasB[samePosMask])

    return np.transpose(np.dstack([pOs, pNs]), axes=[0, 2, 1])


def cutWithPlanesMP(cpPack):
    cellSector, onKeys = cpPack
    for (o, n), cutPlaneKey in onKeys:
        if cutPlaneKey <= 0:
            cellSector.clipWithPlane(o, n, cutPlaneKey)
        else:
            cellSector.cutWithPlane(o, n, cutPlaneKey)
    cellSector.computePolysCentroidsAndWeights()
    return cellSector


def clipCellGeometryMP(csPack):
    cellSec, sitePack, sitesB, MvecsB, lambdasB, domainExtent = csPack
    siteA, MvecA, lambdaA = sitePack

    polysCentroids = cellSec.getPolysCentroids()
    distsA = np.dot(polysCentroids - siteA, MvecA) / lambdaA
    distsL2A = norm(polysCentroids - siteA)
    msk = np.abs(polysCentroids).max(axis=1) < domainExtent

    for siteB, MvecB, lambdaB in zip(sitesB, MvecsB, lambdasB):
        distsB = (np.dot(polysCentroids - siteB, MvecB.T) / lambdaB).max(axis=1)
        dMsk = distsA < distsB

        # equality is actually undefined
        eqMsk = distsA == distsB
        if np.any(eqMsk):
            # use euclidean distance as fallback
            distsL2B = norm(polysCentroids - siteB)
            dMsk[eqMsk * (distsL2A < distsL2B)] = True
        msk *= dMsk

    cellSec.setPolyIoLabels(msk)
    return cellSec


def dissolveMP(cellSectors):
    if len(cellSectors) == 4:
        cVerts = concatPolyParts(flatten([cellSec.getHullVerts() for cellSec in cellSectors]))
        cVertsClean = [limitedDissolve2D(cvs) for cvs in cVerts]
        return cVertsClean, flatten([cellSec.hullPlaneKeys for cellSec in cellSectors])
    elif len(cellSectors) == 6:
        dCell = DissCell(cellSectors)
        dCell.dissolve()
        return dCell


def mpInit(fun, iterData):
    global f, iData
    f, iData = fun, iterData


def mpFun(idx):
    return f(iData[idx])


class ChebyshevObject(ABC):

    @abstractmethod
    def __init__(self, sites, oriFun=None, aniFun=None, extent=1, withMP=True, tag='', nDim=0):
        self.nDim = nDim

        global SectorCutObject
        if self.nDim == 2:
            SectorCutObject = TriCutObject
        elif self.nDim == 3:
            SectorCutObject = PyraCutObject
        else:
            warnings.warn('Currently only 2D and 3D supported.')
            return

        self.sites = sites
        self.numSites = len(self.sites)
        self.sIdxs = range(self.numSites)
        self.colors = np.uint8([hex2rgb(seed2hex(sIdx)) for sIdx in range(self.numSites)])
        self.sitesNeighbors = [[[] for i in range(self.nDim*2)] for sIdx in range(self.numSites)]

        self.oriFun = oriFun
        self.aniFun = aniFun

        self.domainExtent = extent
        self.domainPlanes = np.repeat(np.vstack([np.eye(self.nDim), -np.eye(self.nDim)]), 2, axis=0).reshape(self.nDim*2,2,self.nDim) * [[[self.domainExtent], [1]]]
        self.boxVerts = SectorCutObject.initCellVerts[1:]

        self.withMP = withMP and cpuCount > 1
        self.mpPool = None
        self.mpDesc = '@%d'%cpuCount if self.withMP else ''

        sDiv = np.power(self.numSites, 1.0/self.nDim)
        isGrid = np.isclose(np.rint(sDiv), sDiv) and norm(generateGridPoints(int(np.rint(sDiv)), self.nDim, self.domainExtent) - self.sites).mean() < 1/self.nDim
        self.cellScale = 2 * self.domainExtent / (sDiv if isGrid else 1)
        self.cellScales = np.ones(self.numSites) * self.cellScale

        self.tag = tag if tag else 'infinityVoronoi%dD'%self.nDim
        self.log = Logger('%dD_'%self.nDim + self.tag)
        self.log.logThis('Name', self.tag)
        self.log.logThis('Space', '%dD'%self.nDim)
        self.log.logThis('Sites', self.numSites)
        self.log.logThis('OriFun', self.oriFun is not None)
        self.log.logThis('AniFun', self.aniFun is not None)
        self.log.logThis('CPUs', cpuCount if self.withMP else 1)

        self.timings = []
        self.initCells()

    def initCells(self):
        st = time()

        if self.oriFun is None:
            self.Ms = np.float32([np.eye(self.nDim)] * self.numSites)
        else:
            self.Ms = np.float64(list(map(Mr, self.oriFun(self.sites))))
        self.Mvecs = np.float64([np.vstack([M.T, -M.T]) for M in self.Ms])

        if self.aniFun is None:
            self.lambdas = np.ones((self.numSites, self.nDim*2), np.float32)
        else:
            self.lambdas = self.aniFun(self.sites)

        # geometry (o, n) and topology (sIdx, sJdx)
        self.cutPlanes = {-k: [self.domainPlanes[k], []] for k in range(self.nDim*2)}
        self.cutPlaneKeys = [[[] for j in range(self.nDim*2)] for i in range(self.numSites)]

        self.cellSectors = [[] for sIdx in range(self.numSites)]
        for sIdx in range(self.numSites):
            for di in range(self.nDim*2):
                self.cellSectors[sIdx].append(SectorCutObject(self.sites[sIdx], di, self.cellScales[sIdx] * self.lambdas[sIdx], self.Ms[sIdx]))

                csMins = self.cellSectors[sIdx][-1].vertices.min(axis=0)
                csMaxs = self.cellSectors[sIdx][-1].vertices.max(axis=0)
                for d in range(self.nDim):
                    if csMaxs[d] > self.domainExtent:
                        self.cutPlaneKeys[sIdx][di].append(-d)
                    if csMins[d] < -self.domainExtent:
                        self.cutPlaneKeys[sIdx][di].append(-(d + self.nDim))

        self.timings.append([time() - st])

    def logStats(self, latestOnly=True):
        tms = np.float32(self.timings[-1]) if latestOnly else np.sum(self.timings, axis=0)
        self.log.logThis('Stats', ['Init', 'Ps', 'Cut', 'Clip', 'Diss', 'Total'], ','.join(['% 7s'] * 6))
        self.log.logThis('in s', tms.tolist() + [tms.sum()], ','.join(['%7.2f'] * 6) + 's')
        self.log.logThis('in %', (100 * tms / tms.sum()).tolist() + [tms.sum()], ','.join(['%7.2f'] * 6) + 's')

    def map(self, f, iterData, desc):
        if self.withMP:
            # spawned vs. forked processes
            if True:
                if self.mpPool is None:
                    self.mpPool = mp.Pool(processes=cpuCount)
                mapIterator = self.mpPool.imap(f, iterData)
            else:
                # may avoid os-pipe bottleneck but multiplies mem usage by cpuCount
                if self.mpPool is not None:
                    self.mpPool.close()
                    self.mpPool.join()
                self.mpPool = mp.Pool(processes = cpuCount, initializer = mpInit, initargs = (f, iterData))
                mapIterator = self.mpPool.imap(mpFun, range(len(iterData)))
        else:
            mapIterator = map(f, iterData)

        return list(tqdm(mapIterator, total=len(iterData), ascii=True, desc=desc + self.mpDesc))

    @abstractmethod
    def logMeta(self):
        pass

    def finish(self):
        self.logStats(False)
        if self.mpPool is not None:
            self.mpPool.close()
            self.mpPool.join()
        self.logMeta()

    @abstractmethod
    def getSitesSectorGeometry(self, sIdx, di, sJdx, dj):
        pass

    def computeNeighborsAndPlanes(self):
        siteDimTuples = []
        siteSectorGeometry = []
        for sIdx in tqdm(range(self.numSites), ascii=True, desc='serializing'):
            for sJdx in range(sIdx + 1, self.numSites):
                if not haveCommonElement([sIdx, sJdx], self.sIdxs) or norm(self.sites[sIdx] - self.sites[sJdx]) > np.sqrt(self.nDim) * 2:
                    continue
                for di in range(self.nDim * 2):
                    for dj in range(self.nDim * 2):
                        siteDimTuples.append([sIdx, di, sJdx, dj])
                        siteSectorGeometry.append(self.getSitesSectorGeometry(sIdx, di, sJdx, dj))

        siteDimTuples = np.int32(siteDimTuples)[self.map(sectorsDoIntersectMP, siteSectorGeometry, 'kNN')]

        # vectorized version faster than multiprocessing due to overhead
        ons = computeCutPlanes(
            self.sites[siteDimTuples[:, 0]],
            self.sites[siteDimTuples[:, 2]],
            self.Mvecs[siteDimTuples[:, 0], siteDimTuples[:, 1]],
            self.Mvecs[siteDimTuples[:, 2], siteDimTuples[:, 3]],
            self.lambdas[siteDimTuples[:, 0], siteDimTuples[:, 1]].reshape(-1, 1),
            self.lambdas[siteDimTuples[:, 2], siteDimTuples[:, 3]].reshape(-1, 1))

        # multiprocessing version if available
        #cpPacks = [[self.sites[sIdx], self.sites[sJdx], self.Mvecs[sIdx, di], self.Mvecs[sJdx, dj], self.lambdas[sIdx, di], self.lambdas[sJdx, dj]] for sIdx, di, sJdx, dj in siteDimTuples]
        #ons = self.map(computeCutPlaneMP, cpPacks, 'planes')

        siteDimTuples = [map(int, sdt) for sdt in siteDimTuples]
        for (sIdx, di, sJdx, dj), (o, n) in tqdm(zip(siteDimTuples, ons), total=len(siteDimTuples), ascii=True, desc='addPlanes'):
            self.addCutPlane(sIdx, di, sJdx, dj, o, n)

        if self.nDim == 3: # experimental: register pyra faces as hull
            for cpKey in self.cutPlanes.keys():
                on, sIJdxs = self.cutPlanes[cpKey]
                for sIdx in range(self.numSites):
                    if sIdx not in sIJdxs:
                        continue
                    for di in range(self.nDim*2):
                        for fpKey in range(-10, -5):
                            if planesEquiv(on, self.cellSectors[sIdx][di].facesPlanes[fpKey]):
                                self.cellSectors[sIdx][di].hullKeys.add(fpKey)

    def addCutPlane(self, sIdx, di, sJdx, dj, o, n):
        self.sitesNeighbors[sIdx][di].append((sJdx, dj))
        self.sitesNeighbors[sJdx][dj].append((sIdx, di))

        if norm(n) < eps:
            return

        cutPlaneKey = cantorPi(cantorPiO(sIdx, di), cantorPiO(sJdx, dj))
        self.cutPlanes[cutPlaneKey] = [[o, n], [sIdx, sJdx]]

        for sKdx, dk in [[sIdx, di], [sJdx, dj]]:
            for cpKey in self.cutPlaneKeys[sKdx][dk]:
                if planesEquiv(self.cutPlanes[cpKey][0], [o,n]):
                    break
            else:
                self.cutPlaneKeys[sKdx][dk].append(cutPlaneKey)

    def cutWithPlanes(self):
        cpPacks = []
        for sIdx in self.sIdxs:
            for di in range(self.nDim*2):
                cpPacks.append([self.cellSectors[sIdx][di], [[self.cutPlanes[cpKey][0], cpKey] for cpKey in self.cutPlaneKeys[sIdx][di]]])

        cutResults = self.map(cutWithPlanesMP, cpPacks, 'cutting')
        for sIdx in self.sIdxs:
            for di in range(self.nDim*2):
                self.cellSectors[sIdx][di] = cutResults.pop(0)

    def clipCellGeometry(self):
        csPacks = []
        for sIdx in self.sIdxs:
            for di in range(self.nDim*2):
                sJdxs = np.unique([sJdx for sJdx, dj in self.sitesNeighbors[sIdx][di] if sJdx != sIdx])
                sitePack = [self.sites[sIdx], self.Mvecs[sIdx,di], self.lambdas[sIdx,di]]
                csPacks.append((self.cellSectors[sIdx][di], sitePack, self.sites[sJdxs], self.Mvecs[sJdxs], self.lambdas[sJdxs], self.domainExtent))

        clipResults = self.map(clipCellGeometryMP, csPacks, 'clipping')
        for sIdx in self.sIdxs:
            for di in range(self.nDim*2):
                self.cellSectors[sIdx][di] = clipResults.pop(0)

    @abstractmethod
    def processDissolvedSectors(self):
        pass

    def dissolveCells(self):
        self.processDissolvedSectors(self.map(dissolveMP, self.cellSectors, 'dissolving'))

        self.cellCentroids = np.zeros((self.numSites, self.nDim), np.float32)
        self.cellBBs = np.zeros((self.numSites, 2**self.nDim, self.nDim), np.float32)
        self.cellBBcenters = np.zeros((self.numSites, self.nDim), np.float32)
        self.cellAdjacency = [[] for sIdx in range(self.numSites)]
        for sIdx in tqdm(self.sIdxs, total=len(self.sIdxs), ascii=True, desc='cellData'):

            # L2 centroids
            polyCentroids = np.vstack([cellSec.getPolysCentroids() for cellSec in self.cellSectors[sIdx]])
            polyWeights = np.concatenate([cellSec.getPolysWeights() for cellSec in self.cellSectors[sIdx]])
            self.cellCentroids[sIdx] = np.dot(polyWeights, polyCentroids) / polyWeights.sum()

            # bounding boxes and BB centers
            cellVerts = np.dot(np.vstack(self.cellVertexSets[sIdx]), self.Ms[sIdx])
            minMax = np.float32([cellVerts.min(axis=0), cellVerts.max(axis=0)])
            bbCenter = minMax.mean(axis=0)
            self.cellBBcenters[sIdx] = np.dot(bbCenter, self.Ms[sIdx].T)
            self.cellBBs[sIdx] = np.dot(self.boxVerts * np.abs(minMax[1] - minMax[0]) / 2 + bbCenter, self.Ms[sIdx].T)

            # site indices of adjacent cells
            adjacentSites = flatten([self.cutPlanes[planeKey][1] for planeKey in self.cellPlaneKeys[sIdx] if planeKey in self.cutPlanes.keys()])
            self.cellAdjacency[sIdx] = np.unique([sJdx for sJdx in adjacentSites if sJdx != sIdx])

        self.cellAdjacencyEdges = filterForUniqueEdges(flatten([[[sIdx, aIdx] for aIdx in self.cellAdjacency[sIdx]] for sIdx in self.sIdxs]))

    def computeDiagram(self, finish=True):
        for fun in [self.computeNeighborsAndPlanes, self.cutWithPlanes, self.clipCellGeometry, self.dissolveCells]:
            st = time()
            fun()
            self.timings[-1].append(time() - st)

        if finish:
            self.finish()
        else:
            self.logStats()

    def lloydRelax(self, itersThresh=0, export=False):
        if not hasattr(self, 'cellCentroids'):
            self.computeDiagram(False)

        iterCount = 0
        while True:
            if export:
                self.exportToObj('lr%04d_%s.obj'%(iterCount, self.tag))

            moveDists = norm(self.sites - self.cellCentroids)
            self.log.logThis('Iter', (iterCount, int(itersThresh), moveDists.max(), (type(itersThresh) == float) * itersThresh), '% 4d / % 4d, MD: %0.6f > %0.6f')
            if (itersThresh < 1 and moveDists.max() < itersThresh) or iterCount == itersThresh:
                break

            # use L2 centroids not BB centers
            self.sites = self.cellCentroids.copy()
            self.initCells()
            self.computeDiagram(False)
            iterCount += 1

        self.finish()