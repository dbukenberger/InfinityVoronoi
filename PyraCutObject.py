from util import *


#     6-----7
#    /|    /|
#   5-+---8 |
#   | 2---+-3
#   |/    |/
#   1-----4

initCellVerts = np.vstack([[0, 0, 0], cubeVerts])
dirPoints = np.float32([[-1, 0, -1], [0, 1, -1], [1, 0, -1], [0, -1, -1], [-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, 0, 1], [0, 1, 1], [1, 0, 1], [0, -1, 1]])
faceOrigins = np.vstack([np.eye(3), -np.eye(3), dirPoints/2])
faceNormals = np.vstack([np.eye(3), -np.eye(3), normVec(dirPoints)])

#                        +x                +y                +z                -x                -y               -z
pyraIdxs = [[ 0, 3, 7, 8, 4], [ 0, 2, 6, 7, 3], [ 0, 6, 5, 8, 7], [ 0, 1, 5, 6, 2], [ 0, 4, 8, 5, 1], [ 0, 1, 2, 3, 4]]
foIdxs =   [[ 8,12,16,13, 0], [ 7,11,15,12, 1], [15,14,17,16, 2], [ 6,10,14,11, 3], [ 9,13,17,10, 4], [ 9, 6, 7, 8, 5]]
fnIdxs =   [[ 6,11,14,10, 0], [ 9,10,17,13, 1], [ 7, 6, 9, 8, 2], [ 8,13,16,12, 3], [ 7,12,15,11, 4], [17,14,15,16, 5]]

#          2
#        //|
#      / / |
#    /_,3  |
#   0'--+--1
#    \  | /
#      \|/
#       4

pyraFaces = [[0, 1, 4], [0, 2, 1], [0, 3, 2], [0, 4, 3], [1, 2, 3, 4]]
pyraFaceMaps = {0: 53, 1: 13, 2: 23, 3: 41, 10: 52, 11: 33, 12: 20, 13: 1, 20: 12, 21: 32, 22: 42, 23: 2, 30: 51, 31: 43, 32: 21, 33: 11, 40: 50, 41: 3, 42: 22, 43: 31, 50: 40, 51: 30, 52: 10, 53: 0}


class PyraCutObject:

    initCellVerts = initCellVerts

    def __init__(self, site, di, scale, M, vertsAndFaces=None):

        self.site = site
        self.di = di
        self.scale = scale
        self.M = M

        if di in [0, 3]:
            vertScales = [[1, 1, 1], [scale[di], scale[di+1], scale[5]], [scale[di], scale[di+1], scale[2]], [scale[di], scale[(di+4) % 6], scale[2]], [scale[di], scale[(di+4) % 6], scale[5]]]
            poScales = [[scale[di], 1, scale[5]], [scale[di], scale[di+1], 1], [scale[di], 1, scale[2]], [scale[di], scale[(di+4) % 6], 1], [scale[di], 1, 1]]
            pnScales = [[scale[di], 1, scale[5]], [scale[di], scale[di+1], 1], [scale[di], 1, scale[2]], [scale[di], scale[(di+4) % 6], 1], [1, 1, 1]]

        elif di in [1, 4]:
            vertScales = [[1, 1, 1], [scale[(di+2) % 6], scale[di], scale[5]], [scale[(di+2) % 6], scale[di], scale[2]], [scale[di-1], scale[di], scale[2]], [scale[di-1], scale[di], scale[5]]]
            poScales = [[1, scale[di], scale[5]], [scale[(di+2) % 6], scale[di], 1], [1, scale[di], scale[2]], [scale[(di+2) % 6], scale[di], 1], [1, scale[di], 1]]
            pnScales = [[1, scale[di], scale[5]], [scale[(di+2) % 6], scale[di], 1], [1, scale[di], scale[2]], [scale[di-1], scale[di], 1], [1, 1, 1]]

        elif di in [2, 5]:
            vertScales = [[1, 1, 1], [scale[3], scale[di-1], scale[di]], [scale[3], scale[(di+2) % 6], scale[di]], [scale[0], scale[(di+2) % 6], scale[di]], [scale[0], scale[di-1], scale[di]]]
            poScales = [[1, scale[di-1], scale[di]], [scale[3], 1, scale[di]], [1,scale[(di+2) % 6], scale[di]], [scale[0], 1, scale[di]], [1, 1, scale[di]]]
            pnScales = [[1, scale[di-1], scale[di]], [scale[3], 1, scale[di]], [1, scale[(di+2) % 6], scale[di]], [scale[0], 1, scale[di]], [1, 1, 1]]

        if vertsAndFaces is None:
            self.vertices = site + np.dot(self.initCellVerts[pyraIdxs[di]] * vertScales, M.T)
            self.pyraVertices = self.vertices.copy()
            self.faces = list(map(np.int32, pyraFaces))
        else:
            self.vertices, self.faces = vertsAndFaces

        fOrigins = self.site + np.dot(faceOrigins[foIdxs[di]] * poScales, self.M.T)
        fNormals = np.dot(normVec(faceNormals[fnIdxs[di]]/pnScales), self.M.T)

        self.facesPlanes = {-(i + 6): [fOrigins[i], fNormals[i]] for i in range(5)}
        self.facePlaneKeys = [-6, -7, -8, -9, -10]
        self.hullKeys = set([0, -1, -2, -3, -4, -5])

        self.facePolyIdxs = [np.int64([1, -1]) for f in self.faces]
        self.polys = {1: list(range(len(self.faces)))}

        self.nPolys = [1]

    def clipWithPlane(self, o, n, cutPlaneKey):
        if len(self.polys) > 1:
            warnings.warn('Clipping is supported only on unfractured geometry.')

        dots = np.dot(self.vertices - o, n)
        inside = dots < -eps
        inPlane = np.abs(dots) < eps

        if np.all(inside) or not np.any(inside):
            self.nPolys.append(len(self.polys))
            return False

        if np.any(inPlane):
            if np.all(inside[inPlane ^ True]) or not np.any(inside[inPlane ^ True]):
                self.nPolys.append(len(self.polys))
                return False

        newVertices = self.vertices.tolist()
        newFaces = []
        newFacesPlaneKeys = []
        newEdges = []
        newEdgesHashs = []
        edgeReplaced = {}
        edgesToCut = []

        planeKeysToPop = []
        for fIdx, face in enumerate(self.faces):

            fInsideSum = inside[face].sum()
            if fInsideSum == len(face):
                # face completely inside - take it and continue
                newFaces.append(face)
                newFacesPlaneKeys.append(self.facePlaneKeys[fIdx])
                continue
            if not fInsideSum:
                # face completely outside - throw away and continue
                planeKeysToPop.append(self.facePlaneKeys[fIdx])
                continue

            newFace = []
            newEdge = []

            curIn = inside[face[-1]]
            cutEdge = False

            for i, uIdx in enumerate(face):

                if curIn:
                    # we think we are inside
                    if inside[uIdx]:
                        # we actually are - take the idx
                        newFace.append(uIdx)
                    else:
                        # we stepped outside - cut the edge
                        cutEdge = True
                        curIn = False
                else:
                    # we think we are outside
                    if inside[uIdx]:
                        # we stepped inside - cut the edge
                        cutEdge = True
                        curIn = True
                    else:
                        # we actually are - throw away the idx
                        continue

                if cutEdge:
                    vIdx = face[i-1]
                    eKey = (uIdx, vIdx) if vIdx > uIdx else (vIdx, uIdx)

                    if not eKey in edgeReplaced.keys():

                        if inPlane[uIdx] and inPlane[vIdx]:
                            edgeReplaced[eKey] = uIdx if uIdx in edgeReplaced.values() else vIdx
                        elif inPlane[uIdx]:
                            edgeReplaced[eKey] = uIdx
                        elif inPlane[vIdx]:
                            edgeReplaced[eKey] = vIdx
                        else:
                            newVertex = intersectEdgePlane(self.vertices[[uIdx,vIdx]], o, n)
                            sameVertIdx = np.where(norm(np.float32(newVertices) - newVertex) < eps)[0]
                            if sameVertIdx.size:
                                edgeReplaced[eKey] = sameVertIdx[0]
                            else:
                                edgeReplaced[eKey] = len(newVertices)
                                newVertices.append(newVertex)

                    edgeReplacedIdx = edgeReplaced[eKey]
                    appendUnique(newEdge, edgeReplacedIdx)
                    appendUnique(newFace, edgeReplacedIdx)

                    if curIn and uIdx != edgeReplacedIdx:
                        appendUnique(newFace, uIdx)

                    cutEdge = False

            if len(newEdge) == 2 and newEdge[0] != newEdge[1]:
                newEdgeHash = cantorPi(newEdge[0], newEdge[1])
                if newEdgeHash not in newEdgesHashs:
                    newEdges.append(newEdge)
                    newEdgesHashs.append(newEdgeHash)

            if len(newFace) > 2:
                if newFace[0] == newFace[-1]:
                    newFace = newFace[:-1]
                newFaces.append(newFace)
                newFacesPlaneKeys.append(self.facePlaneKeys[fIdx])

        if len(newEdges) > 2:
            for newFace in edgesToPaths(newEdges):
                newFaces.append(newFace)
                newFacesPlaneKeys.append(cutPlaneKey)

        vIdxs = np.unique(np.concatenate(newFaces))
        self.vertices = np.float32(newVertices)[vIdxs]
        self.faces = reIndexIndices(newFaces)

        self.facePlaneKeys = newFacesPlaneKeys
        if not cutPlaneKey in self.facesPlanes.keys():
            self.facesPlanes[cutPlaneKey] = [o, n]

        for fpKey in planeKeysToPop:
            self.facesPlanes.pop(fpKey)

        self.facePolyIdxs = [np.int64([1, -1]) for f in self.faces]
        self.polys = {1: list(range(len(self.faces)))}

    def cutWithPlane(self, o, n, cutPlaneKey):
        dots = np.dot(self.vertices - o, n)
        inside = dots < -eps
        inPlane = np.abs(dots) < eps

        if np.all(inside) or not np.any(inside):
            self.nPolys.append(len(self.polys))
            return False

        if np.any(inPlane):
            if np.all(inside[inPlane ^ True]) or not np.any(inside[inPlane ^ True]):
                self.nPolys.append(len(self.polys))
                return False

        faceMasks = [inside[face] for face in self.faces]
        inPlaneMasks = [inPlane[face] for face in self.faces]

        if np.any([2 < inPlaneMask.sum() for inPlaneMask in inPlaneMasks]):
            return False

        numVerts = len(self.vertices)
        newFaces = []
        newFacePolyIdxs = []
        newFacesPlaneKeys = []

        edgeReplaced = {}
        edgeReplacedPlaneKeys = {}
        edgesToCut = []

        newEdges = {}
        cutPolys = set(flatten([self.facePolyIdxs[fIdx] for fIdx, fm in enumerate(faceMasks) if any(fm) and not all(fm)]))

        cutFaces = []
        cutFacesMasks = []
        cutFacePolyIdxs = []
        cutFacesPlaneKeys = []
        for faceMask, inPlaneMask, face, fpi, fpKey in zip(faceMasks, inPlaneMasks, self.faces, self.facePolyIdxs, self.facePlaneKeys):

            # fix inPlane false-positives - avoid face cuts with [v, v] edges
            if any(inPlaneMask) and not all(inPlaneMask):

                # in-plane edges of a cutPoly but not a cutFace
                ipEdges = [inPlaneMask[i] and inPlaneMask[(i+1) % len(face)] for i in range(len(face))]
                if any(ipEdges):
                    newEdge = face[inPlaneMask]
                    if len(newEdge) > 2:
                        print('oh oh, inPlane edge hack')
                        newEdge = newEdge[[0, -1]]

                    for i in fpi:
                        if i in cutPolys:
                            pk = (i*2, i*2+1)
                            if not pk in newEdges.keys():
                                newEdges[pk] = []
                            newEdges[pk].append(newEdge.tolist())

                if all(faceMask[inPlaneMask ^ True] == 0):
                    faceMask[inPlaneMask] = 0
                elif all(faceMask[inPlaneMask ^ True] == 1):
                    faceMask[inPlaneMask] = 1

            if not any(faceMask):
                newFaces.append(face)
                newFacesPlaneKeys.append(fpKey)
                if fpi[0] in cutPolys and fpi[1] in cutPolys:
                    newFacePolyIdxs.append(fpi * 2)
                elif fpi[0] in cutPolys:
                    newFacePolyIdxs.append(fpi * [2, 1])
                elif fpi[1] in cutPolys:
                    newFacePolyIdxs.append(fpi * [1, 2])
                else:
                    newFacePolyIdxs.append(fpi)
            elif all(faceMask):
                newFaces.append(face)
                newFacesPlaneKeys.append(fpKey)
                if fpi[0] in cutPolys and fpi[1] in cutPolys:
                    newFacePolyIdxs.append(fpi * 2 + 1)
                elif fpi[0] in cutPolys:
                    newFacePolyIdxs.append(fpi * [2, 1] + [1, 0])
                elif fpi[1] in cutPolys:
                    newFacePolyIdxs.append(fpi * [1, 2] + [0, 1])
                else:
                    newFacePolyIdxs.append(fpi)
            else:
                cutFaces.append(face)
                cutFacesMasks.append(faceMask)
                cutFacePolyIdxs.append(fpi)
                cutFacesPlaneKeys.append(fpKey)


        cutIdxs = computeFaceCutIdxs(cutFacesMasks) if len(cutFacesMasks) else []
        for idxs, cutFaceMask, face, fpi, fpKey in zip(cutIdxs, cutFacesMasks, cutFaces, cutFacePolyIdxs, cutFacesPlaneKeys):
            cutVerts = face[[idxs[0], idxs[0]-1, idxs[1], idxs[1]-1]].reshape(2, 2)

            newEdge = []
            for uIdx, vIdx in cutVerts:
                eKey = (uIdx, vIdx) if vIdx > uIdx else (vIdx, uIdx)

                if eKey not in edgeReplaced.keys():
                    if inPlane[uIdx]:
                        edgeReplaced[eKey] = uIdx
                    elif inPlane[vIdx]:
                        edgeReplaced[eKey] = vIdx
                    else:
                        edgeReplaced[eKey] = numVerts
                        numVerts += 1
                        edgesToCut.append(eKey)
                        edgeReplacedPlaneKeys[eKey] = set([fpKey])
                if eKey in edgeReplacedPlaneKeys.keys() and len(edgeReplacedPlaneKeys[eKey]) < 2:
                    # the two planes containing the replaced edge - intersected later with the third cut plane
                    edgeReplacedPlaneKeys[eKey].add(fpKey)

                newEdge.append(edgeReplaced[eKey])

            assert len(newEdge) == 2 and newEdge[0] != newEdge[1], 'oh oh, this should not happen'

            faceParts = [face[:idxs[0]], face[idxs[1]:], face[idxs[0]:idxs[1]]]

            if inPlane[face].sum():
                faceParts = [facePart[np.bitwise_not(inPlane[facePart])] for facePart in faceParts]

            newFaces.append(np.concatenate([faceParts[0], newEdge, faceParts[1]]))
            newFaces.append(np.concatenate([faceParts[2], newEdge[::-1]]))
            newFacesPlaneKeys += [fpKey, fpKey]

            if haveCommonElement(face[cutFaceMask], newFaces[-2]):
                newFacePolyIdxs.append(fpi*2+1)
                newFacePolyIdxs.append(fpi*2)
            else:
                newFacePolyIdxs.append(fpi*2)
                newFacePolyIdxs.append(fpi*2+1)

            for i in fpi:
                if i > 0:
                    pk = (i*2, i*2+1)
                    if not pk in newEdges.keys():
                        newEdges[pk] = []
                    newEdges[pk].append(newEdge)

        for pk in newEdges.keys():
            newEdgesPK = filterForUniqueEdges(newEdges[pk])
            if len(newEdgesPK) < 3:
                # should actually not happen
                continue

            newFace = edgesToPath(newEdgesPK)
            if newFace is not None:
                newFaces.append(np.int32(newFace))
                newFacePolyIdxs.append(np.int64(pk))
                newFacesPlaneKeys.append(cutPlaneKey)

        nfpis = np.unique(np.concatenate(newFacePolyIdxs))
        self.polys = {nfpi: [] for nfpi in nfpis[nfpis > 0]}
        for faceIdx, (i, j) in enumerate(newFacePolyIdxs):
            if i > 0:
                self.polys[i].append(faceIdx)
            if j > 0:
                self.polys[j].append(faceIdx)

        self.faces = newFaces
        self.facePolyIdxs = newFacePolyIdxs
        self.facePlaneKeys = newFacesPlaneKeys

        if cutPlaneKey not in self.facesPlanes.keys():
            self.facesPlanes[cutPlaneKey] = [o, n]

        if len(edgesToCut):
            #self.vertices = np.vstack([self.vertices, intersectEdgesPlane(self.vertices[np.int32(edgesToCut)], o, n)])
            # theoretically accumulates errors ... use only as fallback solution
            # where the [nI,nJ,n] matrix is singular at razor-blade-like edges

            # per default use three planes to compute vertices
            cutVerts = np.empty((len(edgeReplacedPlaneKeys), 3), np.float32)
            cutEdgeFallback = []
            cutEdgeIdxs = []
            for idx, k in enumerate(edgeReplacedPlaneKeys.keys()):
                piKey, pjKey = edgeReplacedPlaneKeys[k]
                oI, nI = self.facesPlanes[piKey]
                oJ, nJ = self.facesPlanes[pjKey]

                if vecsParallel(nI, nJ) or vecsParallel(nI, n) or vecsParallel(nJ, n):
                    cutEdgeFallback.append(k)
                    cutEdgeIdxs.append(idx)
                else:
                    pOs = np.float32([oI, oJ, o])
                    pNs = np.float32([nI, nJ, n])
                    cutVerts[idx] = intersectThreePlanes(pOs, pNs)

            if cutEdgeIdxs:
                cutVerts[cutEdgeIdxs] = intersectEdgesPlane(self.vertices[np.int32(cutEdgeFallback)], o, n)
            self.vertices = np.vstack([self.vertices, cutVerts])

        self.nPolys.append(len(self.polys))
        return True

    def computePolysCentroidsAndWeights(self):
        if not hasattr(self, 'polysCentroids'):
            self.polysCentroids = {}
        if not hasattr(self, 'polysVolumes'):
            self.polysVolumes = {}
        for pk in self.polys.keys():
            self.polysCentroids[pk], self.polysVolumes[pk] = computePolyhedronCentroid(self.vertices, [self.faces[fIdx] for fIdx in self.polys[pk]], True)

    def getPolysCentroids(self, ioClipped=True):
        if not hasattr(self, 'polysCentroids'):
            self.computePolysCentroidsAndWeights()
        centroids = []
        for pk in self.polys.keys():
            if ioClipped and hasattr(self, 'polysIoLabel') and not self.polysIoLabel[pk]:
                continue
            centroids.append(self.polysCentroids[pk])
        return centroids

    def getPolysWeights(self, ioClipped=True):
        if not hasattr(self, 'polysVolumes'):
            self.computePolysCentroidsAndWeights()
        volumes = []
        for pk in self.polys.keys():
            if ioClipped and hasattr(self, 'polysIoLabel') and not self.polysIoLabel[pk]:
                continue
            volumes.append(self.polysVolumes[pk])
        return volumes

    def getHullVerts(self):
        vIdxs = set()
        for pk in self.polys.keys():
            if hasattr(self, 'polysIoLabel') and not self.polysIoLabel[pk]:
                continue
            vIdxs.update(flatten([self.faces[fIdx] for fIdx in self.polys[pk]]))
        return self.vertices[np.int32(list(vIdxs))]

    def setPolyIoLabels(self, msk):
        if not hasattr(self, 'polysIoLabel'):
            self.polysIoLabel = {pk: True for pk in self.polys.keys()}
            self.facePolyIdxs = np.int64(self.facePolyIdxs)

        self.cellPolyIdxs = []
        for pIdx, (pk, io) in enumerate(zip(self.polys.keys(), msk)):
            self.polysIoLabel[pk] = io
            if io:
                self.cellPolyIdxs.append(pIdx)
            else:
                self.facePolyIdxs[self.facePolyIdxs == pk] *= 0

    def plot(self, withVertIdxs=False, withFacePlaneKeys=False, withSolids=False, withPyraPlanes=False):
        if mlabMissing:
            warnings.warn('Mayavi missing.')
            return

        # wireframe
        eTris = toEdgeTris(facesToEdges(self.faces))
        tPlot = mlab.triangular_mesh(self.vertices[:,0], self.vertices[:,1], self.vertices[:,2], eTris, color=(1,1,1), representation='mesh', tube_radius=0.005)

        if withVertIdxs:
            for i, v in enumerate(self.vertices):
                mlab.text3d(v[0], v[1], v[2], str(i), scale=(0.1,0.1,0.1))

        if withFacePlaneKeys:
            for i, face in enumerate(self.faces):
                fc = self.vertices[face].mean(axis=0)
                mlab.text3d(fc[0], fc[1], fc[2], str(self.facePlaneKeys[i]), scale=(0.1,0.1,0.1))

        if withSolids:
            verts, tris = [], []
            vOffset = 0
            for i, pk in enumerate(self.polys.keys()):
                fcs = [self.faces[f] for f in self.polys[pk]]
                vs = self.vertices[np.unique(flatten(fcs))]
                verts.append(vs + (vs.mean(axis=0) - vs) * 0.1)
                tris.append(facesToTris(reIndexIndices(fcs)) + vOffset)
                vOffset += len(vs)

            x, y, z = np.vstack(verts).T
            scals = np.repeat(np.arange(len(self.polys)), list(map(len, verts)))
            sPlot = mlab.triangular_mesh(x, y, z, np.vstack(tris), scalars=scals, representation='surface')
            sPlot.module_manager.scalar_lut_manager.lut.table = rgb2rgba([hex2rgb(seed2hex(i)) for i in range(len(self.polys))])

        if withPyraPlanes:
            pOs, pNs = zip(*[self.facesPlanes[-k] for k in range(6,11) if -k in self.facePlaneKeys])
            x, y, z = np.float32(pOs).T
            u, v, w = np.float32(pNs).T
            mlab.quiver3d(x, y, z, u, v, w)

        mlab.show()

    def getHullData(self, withPlaneKeys=False):
        if not hasattr(self, 'polysCentroids'):
            self.computePolysCentroidsAndWeights()

        faces = []
        fpIds = []
        fpKeys = []
        for face, fpi, fpKey in zip(self.faces, self.facePolyIdxs, self.facePlaneKeys):
            facePolyIdxsSignedSum = simpleSign(fpi).sum()
            if facePolyIdxsSignedSum == 1 or (fpKey in self.hullKeys and facePolyIdxsSignedSum == 0):
                faces.append(face)
                fpIds.append(max(fpi))
                fpKeys.append(fpKey)

        if not len(faces) or max(fpKeys) <= -6:  # cell in init state
            faces = [self.faces[-1]]
            fpKeys = [(self.facePlaneKeys[-1]-1) * (self.di+1)]

        verts = self.vertices[np.unique(flatten(faces))]
        faces = reIndexIndices(faces)

        if len(fpIds):
            orders = [computeConvexPolygonVertexOrder(verts[face], self.polysCentroids[fpi]) for face, fpi in zip(faces, fpIds)]
            faces = [face[order] for face, order in zip(faces, orders)]

        return (verts, faces, fpKeys) if withPlaneKeys else (verts, faces)

    def getObjData(self, separatedPolys=True, ioClipped=True):
        if separatedPolys and not hasattr(self, 'polysCentroids'):
            self.computePolysCentroidsAndWeights()

        verts, faces = [], []
        vOffset = 0
        for pk in self.polys.keys():
            if ioClipped and hasattr(self, 'polysIoLabel') and not self.polysIoLabel[pk]:
                continue
            else:
                polyFaces = [self.faces[fIdx] for fIdx in self.polys[pk]]

                if separatedPolys:
                    vs = self.vertices[np.unique(flatten(polyFaces))]
                    polyFaces = reIndexIndices(polyFaces)

                    verts.append(vs)
                    c = self.polysCentroids[pk]
                    for f in polyFaces:
                        o = computeConvexPolygonVertexOrder(vs[f], c)
                        faces.append(f[o] + vOffset)
                    vOffset += len(vs)
                else:
                    faces += polyFaces

        return (np.vstack(verts), faces) if separatedPolys else (self.vertices[np.unique(flatten(faces))], reIndexIndices(faces))

    def writeToObj(self, filePath='pyra.obj', separatedPolys=True, ioClipped=True):
        verts, faces = self.getObjData(separatedPolys, ioClipped)
        writeObjFile(filePath, verts, faces)