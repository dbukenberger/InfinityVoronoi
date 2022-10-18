from util import *
from PyraCutObject import *


class DissCell:

    def __init__(self, sixCellPyras):
        self.verts, self.faces, self.facesPlaneDiKeyTuples = [], [], []
        vOffset = 0
        for di, cellPyra in enumerate(sixCellPyras):
            vs, fs, fpKeys = cellPyra.getHullData(True)
            self.verts.append(vs)
            self.faces += [(f + vOffset).tolist() for f in fs]
            vOffset += len(vs)
            self.facesPlaneDiKeyTuples += [(di, fpKey) for fpKey in fpKeys]
        self.verts = np.vstack(self.verts)

        self.facesEdgeHashs = [[] for face in self.faces]
        self.computeFacesEdgeHashs()

    def dissolve(self):
        if min(list(zip(*self.facesPlaneDiKeyTuples))[1]) in [-6,-7,-8,-9,-10]:
            self.clearInnerFaces()

        self.dissolveFaces()
        self.removeDuplicateVerts()
        self.dissolveEdges()
        self.dissolveVerts()
        # check if second pass is required?

        self.dissolveFaces(ignoreDis=True)
        self.dissolveEdges()
        self.removeDuplicateVerts()

    def computeFacesEdgeHashs(self, fIdxs=[]):
        fIdxs = range(len(self.faces)) if not len(fIdxs) else fIdxs
        for fIdx in fIdxs:
            es = faceToEdges(self.faces[fIdx])
            self.facesEdgeHashs[fIdx] = set(cantorPiV(es))

    def clearInnerFaces(self):
        fIdxs, faces, fpMaps = map(list, zip(*[(fIdx, face, fpKeyTup[0]*10-fpKeyTup[1]-6) for fIdx, (face, fpKeyTup) in enumerate(zip(self.faces, self.facesPlaneDiKeyTuples)) if fpKeyTup[1] <= -6]))
        fCenters = map(lambda face: self.verts[face].mean(axis=0), faces)

        fpMapIdxs = np.concatenate([[fpMap]*(len(face)-2) for fpMap, face in zip(fpMaps, faces)])
        fTriMasks = map(lambda fpMap: fpMapIdxs == pyraFaceMaps[fpMap], fpMaps)

        tris = facesToTris(faces)
        ABCs = self.verts[tris]
        uvws = ABCs[:,[1,2,0]] - ABCs
        ns = np.transpose(np.dstack([cross(uvws[:,i], -uvws[:,(i+2)%3], True) for i in range(3)]), axes = [0,2,1])

        self.removeFaces([fIdx for fIdx, fCenter, m in zip(fIdxs, fCenters, fTriMasks) if pointInTriangles3D(ABCs[m], fCenter, uvws[m], ns[m], True)])

    def removeFaces(self, fIdxs):
        for fIdx in sorted(np.unique(fIdxs))[::-1]:
            self.removeFace(fIdx)

    def removeFace(self, fIdx):
        self.faces.pop(fIdx)
        self.facesEdgeHashs.pop(fIdx)
        self.facesPlaneDiKeyTuples.pop(fIdx)

    def mergeFaces(self, fIdx, fJdx):
        fIdx, fJdx = min(fIdx, fJdx), max(fIdx, fJdx)

        idges = faceToEdges(self.faces[fIdx])
        jdges = faceToEdges(self.faces[fJdx])
        edges = filterForSingleEdges(np.vstack([idges, jdges]))

        vIdxs = edges.ravel()
        if len(np.unique(vIdxs)) != len(vIdxs)//2:
            return False

        newFace = edgesToPath(edges)
        if newFace is None:
            return False

        self.faces[fIdx] = newFace
        self.computeFacesEdgeHashs([fIdx])

        self.removeFace(fJdx)

        return True

    def areFacesCoplanar(self, fIdx, fJdx, ignoreDis=False):
        if ignoreDis:
            if self.facesPlaneDiKeyTuples[fIdx][1] <= 0 and self.facesPlaneDiKeyTuples[fJdx][1] <= 0:
                return self.facesPlaneDiKeyTuples[fIdx][1] == self.facesPlaneDiKeyTuples[fJdx][1]
        return self.facesPlaneDiKeyTuples[fIdx] == self.facesPlaneDiKeyTuples[fJdx]

    def dissolveFaces(self, ignoreDis=False):
        iLast = 0
        while True:
            for fIdx in range(iLast, len(self.faces)):
                iLast = fIdx
                iHashs = self.facesEdgeHashs[fIdx]
                merged = False
                for fJdx in range(fIdx+1, len(self.faces)):
                    if not iHashs.isdisjoint(self.facesEdgeHashs[fJdx]):
                        if self.areFacesCoplanar(fIdx, fJdx, ignoreDis):
                            merged = self.mergeFaces(fIdx, fJdx)
                            if merged:
                                break
                if merged:
                    break
            else:
                break

    def removeVertex(self, vIdx):
        self.verts = np.vstack([self.verts[:vIdx], self.verts[vIdx+1:]])

        facesToRemove = []
        for fIdx, face in enumerate(self.faces):
            if vIdx in face:
                face.pop(face.index(vIdx))
                if len(face) < 3:
                    facesToRemove.append(fIdx)
            for i in range(len(face)):
                if face[i] > vIdx:
                    face[i] -= 1
        self.removeFaces(facesToRemove)

    def dissolveEdges(self):
        vRemove = [[] for v in self.verts]
        for face in self.faces:
            n = len(face)
            for i in range(n):
                pdx = face[(i-1) % n]
                idx = face[i]
                ndx = face[(i+1) % n]

                if pdx == ndx:
                    vRemove[idx].append(True)
                    continue

                vP, vI, vN = self.verts[[pdx, idx, ndx]]
                dst = distPointToEdge(vP, vN, vI)
                if dst < eps:
                    vRemove[idx].append(True)
                    continue
                u, v = normVec(self.verts[[pdx, ndx]] - self.verts[idx])
                dt = 1-np.abs(np.dot(u, v))
                if dt < eps:
                    vRemove[idx].append(True)
                    continue

                vRemove[idx].append(max(dst, dt) < eps*100)

        vertsToRemove = [vIdx for vIdx, votes in enumerate(vRemove) if all(votes)]
        for vIdx in sorted(np.unique(vertsToRemove))[::-1]:
            self.removeVertex(vIdx)

    def dissolveVerts(self):
        vIdxs = np.unique(flatten(self.faces))
        self.verts = self.verts[vIdxs]
        self.faces = reIndexIndices(self.faces)
        self.faces = [f[np.nonzero(f-np.roll(f, -1))[0]].tolist() for f in self.faces]
        facesToRemove = [fIdx for fIdx, face in enumerate(self.faces) if len(face) < 3]
        self.removeFaces(facesToRemove)
        self.computeFacesEdgeHashs()

    def removeDuplicateVerts(self, thresh=eps):
        dupliVerts = []
        duplis = set()
        for vIdx, v in enumerate(self.verts[:-1]):
            if vIdx in duplis:
                continue
            ds = norm(self.verts[(vIdx+1):] - v)
            mIdxs = np.where(ds < thresh)[0]
            if len(mIdxs):
                dIdxs = (mIdxs + vIdx + 1).tolist()
                dupliVerts.append((vIdx, dIdxs))
                duplis.update(dIdxs)

        def areNeighbors(face, idx, jdx):
            iPos = face.index(idx)
            jPos = face.index(jdx)
            return max(iPos, jPos) - min(iPos, jPos) in [1, len(face)-1]

        for vIdx, rIdxs in dupliVerts:
            valid = True
            valids = []
            for fIdx, face in enumerate(self.faces):
                directNeighbors = False

                if vIdx in face:
                    for rIdx in rIdxs:
                        if rIdx in face:
                            if areNeighbors(face, rIdx, vIdx):
                                directNeighbors = True
                            else:
                                valid = False
                                break

                for i, rIdx in enumerate(rIdxs):
                    for rJdx in rIdxs[i+1:]:
                        if rIdx in face and rJdx in face:
                            if not areNeighbors(face, rIdx, rJdx):
                                valid = False
                                break

                if not valid:
                    break
                valids.append(directNeighbors)
                if not any(valids):
                    continue
                if not all(valids):
                    break
            else:
                for face in self.faces:
                    for rIdx in rIdxs:
                        if rIdx in face:
                            face[face.index(rIdx)] = vIdx

        self.dissolveVerts()

    def getFacesTriangulated(self):
        return np.vstack([np.int32(face)[triangulatePoly3D(self.verts[face])] for face in self.faces])

    def plot(self, withVertIdxs=False, withFacePlaneKeys=False):
        if mlabMissing:
            warnings.warn("Mayavi missing.")
            return

        # wireframe
        eTris = np.pad(facesToEdges(self.faces), [[0, 0], [0, 1]], "reflect")
        mlab.triangular_mesh(self.verts[:, 0], self.verts[:, 1], self.verts[:, 2], eTris, representation="mesh", tube_radius=0.005)

        if withVertIdxs:
            for i, v in enumerate(self.verts):
                mlab.text3d(v[0], v[1], v[2], str(i), scale=(0.01, 0.01, 0.01))

        if withFacePlaneKeys:
            for i, face in enumerate(self.faces):
                fc = self.verts[face].mean(axis=0)
                mlab.text3d(fc[0], fc[1], fc[2], str(
                    self.facesPlaneDiKeyTuples[i]), scale=(0.01, 0.01, 0.01))

    def writeToObj(self, fName="dissCell.obj"):
        writeObjFile(fName, self.verts, list(map(np.int32, self.faces)))