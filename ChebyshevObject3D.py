from ChebyshevObject import *


class ChebyshevObject3D(ChebyshevObject):

    def __init__(self, sites, oriFun=None, aniFun=None, extent=1, withMP=True, tag=''):
        super().__init__(sites, oriFun, aniFun, extent, withMP, tag, 3)

    def logMeta(self):
        self.log.logThis('#/Cell', ['Min', 'Max', 'Med'], ','.join(['% 4s'] * 3))
        numNeighbors = list(map(len, self.cellAdjacency))
        numParts = [len(findConnectedComponents(facesToEdges(cDiss.faces))) for cDiss in self.cellDiss]
        numFaces = [len(cDiss.faces) for cDiss in self.cellDiss]
        numVerts = [len(cDiss.verts) for cDiss in self.cellDiss]
        for nVals, tag in zip([numNeighbors, numParts, numFaces, numVerts], ['Adj.', 'Parts', 'Faces', 'Verts']):
            self.log.logThis(tag, [min(nVals), max(nVals), np.median(nVals)], ','.join(['% 4d'] * 3))

    def getSitesSectorGeometry(self, sIdx, di, sJdx, dj):
        iVerts = self.cellSectors[sIdx][di].vertices
        jVerts = self.cellSectors[sJdx][dj].vertices
        iNormals = [self.cellSectors[sIdx][di].facesPlanes[-k][1] for k in range(6,11)]
        jNormals = [self.cellSectors[sJdx][dj].facesPlanes[-k][1] for k in range(6,11)]
        return [iVerts, jVerts, iNormals, jNormals]

    def processDissolvedSectors(self, dData):
        self.cellDiss = dData
        self.cellVertexSets = [[cellPyra.getHullVerts() for cellPyra in self.cellSectors[sIdx]] for sIdx in range(self.numSites)]
        self.cellPlaneKeys = [[fpDiKey[1] for fpDiKey in self.cellDiss[sIdx].facesPlaneDiKeyTuples] for sIdx in range(self.numSites)]

    def plot(self, sIdxs=[], withSites=True, withCentroids=True, withAdjacency=True, withInitCells=True, withBBs=True, withCells=True):
        if not mlabFound:
            warnings.warn('Mayavi missing.')
            return

        sIdxs = sIdxs if len(sIdxs) else self.sIdxs

        colorScals = np.arange(len(sIdxs))
        colorTable = rgb2rgba(self.colors[sIdxs] if len(sIdxs) > 1 else np.vstack([self.colors[sIdxs], [[0,0,0]]]))

        # unit cube domain
        cubeEdgeTris = toEdgeTris(facesToEdges(sixCubeFaces))
        x,y,z = (cubeVerts * self.domainExtent).T
        cubePlot = mlab.triangular_mesh(x, y, z, cubeEdgeTris, representation='mesh', color=(1,1,1), tube_radius=0.01)

        # sites, oriented and scaled
        if withSites:
            x,y,z = np.vstack(self.sites[sIdxs].reshape(-1,1,3) + self.Mvecs[sIdxs] * self.lambdas[sIdxs].reshape(-1,6,1) * 0.1).T
            edgesTris = toEdgeTris(np.concatenate([np.int32([[0,3],[1,4],[2,5]]) + i*6 for i in range(len(sIdxs))]))
            sPlot = mlab.triangular_mesh(x, y, z, edgesTris, scalars=np.repeat(colorScals, 6), representation = 'mesh', tube_radius=0.005)
            sPlot.module_manager.scalar_lut_manager.lut.table = colorTable

        # L2 centroids as cell-colored spheres
        if withCentroids:
            x,y,z = self.cellCentroids[sIdxs].T
            s = np.ones_like(x)
            ctsPlot = mlab.quiver3d(x, y, z, s, s, s, scale_factor=0.01, scalars=colorScals, mode='sphere', resolution=6)
            ctsPlot.glyph.color_mode = 'color_by_scalar'
            ctsPlot.glyph.glyph_source.glyph_source.center = [0, 0, 0]
            ctsPlot.glyph.glyph.clamping = False
            ctsPlot.module_manager.scalar_lut_manager.lut.table = colorTable

        # cell adjacency graph as white edges
        if withAdjacency and len(sIdxs) > 1:
            eTris = toEdgeTris([edge for edge in self.cellAdjacencyEdges if edge[0] in sIdxs and edge[1] in sIdxs])
            aPlot = mlab.triangular_mesh(self.sites[:,0], self.sites[:,1], self.sites[:,2], eTris, color=(1,1,1), representation='mesh', tube_radius=0.01)

        # init cells wireframes
        if withInitCells:
            x,y,z = np.concatenate([np.vstack([self.cellSectors[sIdx][5].pyraVertices, self.cellSectors[sIdx][2].pyraVertices[[2,1,4,3]]]) for sIdx in sIdxs]).T
            boxTris = np.int32([[0,1,2],[0,2,3],[0,3,4],[0,4,1],[0,1,5],[0,2,6],[0,3,7],[0,4,8],[0,5,6],[0,6,7],[0,7,8],[0,8,5]])
            tris = np.concatenate([boxTris + i*9 for i in range(len(sIdxs))])
            pPlot = mlab.triangular_mesh(x, y, z, tris, scalars=np.repeat(colorScals, 9), representation='mesh', tube_radius=0.01)
            pPlot.module_manager.scalar_lut_manager.lut.table = colorTable

        # oriented cell bounding box wireframes and centers as cell-colored cubes
        if withBBs:
            x,y,z = self.cellBBcenters[sIdxs].T
            s = np.ones_like(x)
            bbcPlot = mlab.quiver3d(x, y, z, s, s, s, scale_factor=0.01, scalars=colorScals, mode='cube', resolution=6)
            bbcPlot.glyph.color_mode = 'color_by_scalar'
            bbcPlot.glyph.glyph_source.glyph_source.center = [0, 0, 0]
            bbcPlot.glyph.glyph.clamping = False
            bbcPlot.module_manager.scalar_lut_manager.lut.table = colorTable

            x,y,z = np.vstack([self.cellBBs[sIdx] for sIdx in sIdxs]).T
            eTris = np.vstack([cubeEdgeTris + i*8 for i in range(len(sIdxs))])
            bbPlot = mlab.triangular_mesh(x, y, z, eTris, scalars=np.repeat(colorScals, 8), representation='mesh', tube_radius=0.01)
            bbPlot.module_manager.scalar_lut_manager.lut.table = colorTable

        # semi-transparent solid cells
        if withCells:
            verts = []
            tris = []
            colScals = []
            vOffset = 0
            for i, sIdx in enumerate(sIdxs):
                #verts.append(self.cellDiss[sIdx].verts + (self.sites[sIdx] - self.cellDiss[sIdx].verts)*0.05)
                verts.append(self.cellDiss[sIdx].verts)
                tris.append(self.cellDiss[sIdx].getFacesTriangulated() + vOffset)
                vOffset += len(verts[-1])
                colScals += [i] * len(verts[-1])
            x,y,z = np.vstack(verts).T
            cPlot = mlab.triangular_mesh(x, y, z, np.vstack(tris), scalars=colScals, representation='surface', tube_radius=0.01)
            cPlot.module_manager.scalar_lut_manager.lut.table = colorTable * [[1,1,1,0.5]]

    def exportToObj(self, fileName='', withBBs=False, withAdjacency=True, numIter=None):
        verts, faces, tags = [], [], []

        llPrefix = '' if numIter is None else 'lr%04d_'%numIter
        for sIdx in self.sIdxs:
            if fileName:
                verts.append(self.cellDiss[sIdx].verts)
                faces.append(self.cellDiss[sIdx].faces)
                tags.append(llPrefix + 'cell%04d'%sIdx)
                if withBBs:
                    verts.append(self.cellBBs[sIdx])
                    faces.append(sixCubeFaces)
                    tags.append(llPrefix + 'BB%04d'%sIdx)
            else:
                self.cellDiss[sIdx].writeToObj(resDir + llPrefix + 'cell%04d.obj'%sIdx)
                if withBBs:
                    writeObjFile(resDir + llPrefix + 'BB%04d.obj'%sIdx, self.cellBBs[sIdx], sixCubeFaces)

        if fileName:
            writeObjFile(resDir + fileName, verts, faces, subTags = tags)

        if withAdjacency:
            fileName = (fileName[:-4] + '_' if fileName else '') + 'adjacency.obj' 
            writeObjFile(resDir + llPrefix + fileName, self.sites, [], self.cellAdjacencyEdges)

    def writeCellSectorsToObjs(self, sIdxs=[], separatedPolys=True, ioClipped=True):
        sIdxs = sIdxs if len(sIdxs) else range(self.numSites)

        for sIdx in tqdm(sIdxs, total=len(sIdxs), ascii=True, desc='writing'):
            verts, faces, tags = [], [], []
            for di in range(6):
                vs, fs = self.cellSectors[sIdx][di].getObjData(separatedPolys, ioClipped)
                verts.append(vs)
                faces.append(fs)
                tags.append('c%04d_p%02d'%(sIdx, di))

            writeObjFile(resDir + 'secs%04d.obj'%sIdx, verts, faces, subTags=tags)