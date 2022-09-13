from util import *
from TriCutObject import *


# principle bisector plane computation
def computeCutPlaneMP(cpPack):
    A, B, a, b, lambdaA, lambdaB = cpPack
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

    return o, n


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


def cutWithPlanesMP(cpPack):
    cellTri, onKeys = cpPack
    for (o, n), cutPlaneKey in onKeys:
        cellTri.cutWithLine(o, n, cutPlaneKey)
    cellTri.computeCentroids()
    return cellTri


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


class ChebyshevObject2D:

    domainPlanes = np.repeat(np.vstack([np.eye(2), -np.eye(2)]), 2, axis=0).reshape(4, 2, 2)

    def __init__(self, sites, oriFun=None, aniFun=None, extent=1, withMP=True, tag=''):
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

                cpMins = self.cellTris[sIdx][-1].vertices.min(axis=0)
                cpMaxs = self.cellTris[sIdx][-1].vertices.max(axis=0)
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
        # cpPacks = [[self.sites[sIdx], self.sites[sJdx], self.Mvecs[sIdx, di], self.Mvecs[sJdx, dj], self.lambdas[sIdx, di], self.lambdas[sJdx, dj]] for sIdx, di, sJdx, dj in siteDimTuples]
        # ons = list(tqdm(self.mpPool.imap(computeCutPlaneMP, cpPacks, chunksize=cpuCount*4), total=len(cpPacks), ascii=True, desc="planes@%d"%cpuCount))

        for (sIdx, di, sJdx, dj), (o, n) in tqdm(zip(siteDimTuples, ons), total=len(siteDimTuples), ascii=True, desc="addPlanes"):
            self._addCutPlane(sIdx, di, sJdx, dj, o, n)

    def _addCutPlane(self, sIdx, di, sJdx, dj, o, n):
        self.sitesNeighbors[sIdx][di].append((sJdx, dj))
        self.sitesNeighbors[sJdx][dj].append((sIdx, di))

        if norm(n) < eps:
            return

        cutPlaneKey = cantorPi(cantorPiO(sIdx, di), cantorPiO(sJdx, dj))
        self.cutPlanes[cutPlaneKey] = [[o, n], [sIdx, sJdx]]

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
            self.cellBBs[sIdx] = np.dot(initQuadVerts[1:] * np.abs(minMax[1] - minMax[0]) / 2 + bbCenter, self.Ms[sIdx].T)

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
            warnings.warn("matplotlib missing.")
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

    def exportToSvg(self, fileName="infinityVoronoi.svg", size=512, withSites=True):
        theDom = minidom.parseString('<svg xmlns="http://www.w3.org/2000/svg" height="%d" width="%d"></svg>'%(size, size))
        for sIdx in range(self.numSites):
            g = theDom.createElement('g')
            g.setAttribute('style', 'fill:%s;'%rgb2hex(self.colors[sIdx]))
            theDom.childNodes[0].appendChild(g)
            for cellVerts in self.cellVertexSets[sIdx]:
                p = theDom.createElement('path')
                verts = (cellVerts * [1, -1] + 1) / 2 * size
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
    if IN_IDLE:
        # compute the diagram once and plot it
        cObj.computeDiagram()
        cObj.plot()
    else:
        # run the relaxation and save the demo image as SVG
        cObj.lloydRelax(0.001)
        cObj.exportToSvg()
