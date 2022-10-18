from util import *


#   2 - n - 3
#   | \   / |
#   w   0   e
#   | /   \ |
#   1 - s - 4

initCellVerts = np.vstack([[0, 0], quadVerts])
edgeCenters = np.vstack([np.eye(2), -np.eye(2), initCellVerts[1:] * 0.5])
edgeNormals = np.vstack([np.eye(2), -np.eye(2), initCellVerts[1:]])

#           e:   +x    n:   +y    w:   -x    s:   -y
triIdxs = [[0, 3, 4], [0, 2, 3], [0, 1, 2], [0, 4, 1]]
ecIdxs =  [[6, 0, 7], [5, 1, 6], [4, 2, 5], [7, 3, 4]]
enIdxs =  [[5, 0, 4], [4, 1, 7], [7, 2, 6], [6, 3, 5]]


class TriCutObject:

    initCellVerts = initCellVerts

    def __init__(self, site, di, scale, M):

        self.edges = list(map(np.int32, [[0, 1], [1, 2], [2, 0]]))
        self.polys = {1: [0, 1, 2]}
        self.edgePolyIdxs = [np.int64([1, -1]) for e in self.edges]

        vertScales = [[1, 1], [scale[(di + 1) % 4], scale[di]], [scale[di - 1], scale[di]]] if di % 2 else [[1, 1], [scale[di], scale[di + 1]], [scale[di], scale[(di - 1) % 4]]]
        eCenterScales = [[scale[(di + 1) % 4], scale[di]], [1, scale[di]], [scale[(di - 1) % 4], scale[di]]] if di % 2 else [[scale[di], scale[(di + 1) % 4]], [scale[di], 1], [scale[di], scale[(di - 1) % 4]]]
        eNormalScales = [[scale[di], scale[(di + 1) % 4]], [1, 1], [scale[di], scale[(di - 1) % 4]]] if di % 2 else [[scale[(di + 1) % 4], scale[di]], [1, 1], [scale[(di - 1) % 4], scale[di]]]       

        self.vertices = site + np.dot(self.initCellVerts[triIdxs[di]] * vertScales, M.T)
        eCenters = site + np.dot(edgeCenters[ecIdxs[di]] * eCenterScales, M.T)
        eNormals = normVec(np.dot(edgeNormals[enIdxs[di]] * eNormalScales, M.T))

        self.edgesPlanes = {-(i + 4): [eCenters[i], eNormals[i]] for i in range(3)}
        self.edgePlaneKeys = [-4, -5, -6]

    def clipWithPlane(self, o, n, cutPlaneKey):
        # in 2D not so crucial for performance
        self.cutWithPlane(o, n, cutPlaneKey)

    def cutWithPlane(self, o, n, cutPlaneKey):
        dots = np.dot(self.vertices - o, n)
        vMasks = simpleSign(dots, eps)
        onLine = np.abs(dots) < eps

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

            assert len(newEdgeInner) == 2, 'oh oh, this should not happen'

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

    def computePolysCentroidsAndWeights(self):
        self.polysCentroids = np.empty((len(self.polys), 2), np.float32)
        self.polysAreas = np.empty(len(self.polys), np.float32)
        for pIdx, pk in enumerate(self.polys.keys()):
            es = [self.edges[eIdx].tolist() for eIdx in self.polys[pk]]
            self.polysCentroids[pIdx], self.polysAreas[pIdx] = computePolygonCentroid2D(self.vertices[edgesToPath(es)], True)

    def getPolysCentroids(self, ioClipped=True):
        if not hasattr(self, 'polysCentroids'):
            self.computePolysCentroidsAndWeights()
        return self.polysCentroids[self.cellPolyIdxs] if ioClipped and hasattr(self, 'cellPolyIdxs') else self.polysCentroids

    def getPolysWeights(self, ioClipped=True):
        if not hasattr(self, 'polysAreas'):
            self.computePolysCentroidsAndWeights()
        return self.polysAreas[self.cellPolyIdxs] if ioClipped and hasattr(self, 'cellPolyIdxs') else self.polysAreas

    def getHullVerts(self):
        es = {}
        for e, ePolyIdx, ePlaneKey in zip(self.edges, self.edgePolyIdxs, self.edgePlaneKeys):
            if simpleSign(ePolyIdx).sum() == 1:
                if ePlaneKey in es.keys():
                    es[ePlaneKey].append(e)
                else:
                    es[ePlaneKey] = [e]

        if not len(es):  # cell in init state
            self.hullPlaneKeys = [-6]
            return [self.vertices[self.edges[-1]]]

        segs = []
        for epk in es.keys():
            ces = findConnectedEdgeSegments(es[epk])
            for ce in ces:
                ep = edgesToPath(ce)
                segs.append([ep[0], ep[-1]])

        self.hullPlaneKeys = list(es.keys())
        return [self.vertices[seg] for seg in edgesToPaths(segs)]

    def setPolyIoLabels(self, msk):
        if not hasattr(self, 'polysIoLabel'):
            self.polysIoLabel = {pk: True for pk in self.polys.keys()}
            self.edgePolyIdxs = np.int64(self.edgePolyIdxs)

        self.cellPolyIdxs = []
        for pIdx, (pk, io) in enumerate(zip(self.polys.keys(), msk)):
            self.polysIoLabel[pk] = io
            if io:
                self.cellPolyIdxs.append(pIdx)
            else:
                self.edgePolyIdxs[self.edgePolyIdxs == pk] *= 0

    def plot(self):
        if mplMissing:
            warnings.warn('matplotlib missing.')
            return

        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])

        for pKey in self.polys.keys():
            face = edgesToPath([self.edges[eIdx].tolist() for eIdx in self.polys[pKey]])
            cVerts = self.vertices[face]
            cVerts = cVerts - (cVerts - cVerts.mean(axis=0)) * 0.05
            ax.fill(cVerts[:, 0], cVerts[:, 1], fill=self.polysIoLabel[pKey] if hasattr(self, 'polysIoLabel') else False)

        for vIdx, vt in enumerate(self.vertices):
            ax.text(vt[0], vt[1], str(vIdx))

        ax.set_aspect('equal', 'box')
        plt.show()