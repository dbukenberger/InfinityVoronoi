from ChebyshevObject import *


class ChebyshevObject2D(ChebyshevObject):

    def __init__(self, sites, oriFun=None, aniFun=None, extent=1, withMP=True, tag=''):
        super().__init__(sites, oriFun, aniFun, extent, withMP, tag, 2)

    def logMeta(self):
        self.log.logThis('#/Cell', ['Min', 'Max', 'Med'], ','.join(['% 4s'] * 3))
        numNeighbors = list(map(len, self.cellAdjacency))
        numParts = list(map(len, self.cellVertexSets))
        numVerts = list(map(len, map(np.concatenate, self.cellVertexSets)))
        for nVals, tag in zip([numNeighbors, numParts, numVerts], ['Adj.', 'Parts', 'Verts']):
            self.log.logThis(tag, [min(nVals), max(nVals), np.median(nVals)], ','.join(['% 4d'] * 3))

    def getSitesSectorGeometry(self, sIdx, di, sJdx, dj):
        return [self.cellSectors[sIdx][di].vertices, self.cellSectors[sJdx][dj].vertices]

    def processDissolvedSectors(self, dData):
        self.cellVertexSets, self.cellPlaneKeys = list(zip(*dData))

    def plot(self, withSites=True, withCentroids=True, withAdjacency=True, withBBs=True, fileName=''):
        if mplMissing:
            warnings.warn('matplotlib missing.')
            return

        fig = plt.figure('Voronoi Plot')
        ax = fig.add_axes([0, 0, 1, 1])

        legendElements = []

        for sIdx in self.sIdxs:
            for cellVerts in self.cellVertexSets[sIdx]:
                ax.fill(cellVerts[:, 0], cellVerts[:, 1], color=self.colors[sIdx] / 255)

        if withAdjacency:
            adjacencyPlot = ax.plot(self.sites[self.cellAdjacencyEdges, 0].T, self.sites[self.cellAdjacencyEdges, 1].T, color='white', linewidth=1, label='Adjacency')
            legendElements.append(adjacencyPlot[0])

        if withSites:
            dirVecs = np.tile([edgeNormals[:4]], [self.numSites, 1, 1]) * self.lambdas.reshape(-1, 4, 1)
            oriPts = self.sites.reshape(-1, 1, 2) + innerNxM(dirVecs, self.Ms) * self.cellScale / 5
            sitesPlot = ax.plot(np.vstack(oriPts[:, [[0, 2], [1, 3]], 0]).T, np.vstack(oriPts[:, [[0, 2], [1, 3]], 1]).T, color='black')
            legendElements.append(ax.scatter([], [], marker='+', color='black', label='Sites'))

        if withCentroids:
            centroidsPlot = ax.scatter(self.cellCentroids[:, 0], self.cellCentroids[:, 1], color='yellow', s=15, label='$L_2$ Centroids')
            legendElements.append(centroidsPlot)

        if withBBs:
            bbCentersPlot = ax.scatter(self.cellBBcenters[:, 0], self.cellBBcenters[:, 1], color='green', s=15, label='$L_\infty$ BB Centers')
            legendElements.append(bbCentersPlot)
            for sIdx in self.sIdxs:
                ax.plot(self.cellBBs[sIdx][[0, 1, 2, 3, 0], 0], self.cellBBs[sIdx][[0, 1, 2, 3, 0], 1], color=self.colors[sIdx] / 255, linewidth=1)

        ax.set_aspect('equal', 'box')
        fig.legend(handles=legendElements, facecolor='gray')
        plt.axis(False)
        if fileName:
            plt.savefig(resDir + fileName)
        plt.show()

    def exportToObj(self, fileName='', withSites=True, withCentroids=True, withAdjacency=True, withBBs=True, numIter=None):
        llPrefix = '' if numIter is None else 'lr%04d_'%numIter
        fileName = fileName if fileName else self.tag + '.obj'

        numVerts = 0
        verts = []
        cells = []
        for sIdx in self.sIdxs:
            for cellVerts in self.cellVertexSets[sIdx]:
                cells.append(np.arange(len(cellVerts)) + numVerts)
                verts.append(cellVerts)
                numVerts += len(cellVerts)

        vertices = [padHor(np.vstack(verts))]
        faces = [cells]
        edges = [[]]
        tags = [llPrefix + 'cells']

        if withSites:
            vertices.append(padHor(self.sites))
            faces.append([])
            edges.append([])
            tags.append(llPrefix + 'sites')

        if withCentroids:
            vertices.append(padHor(self.cellCentroids))
            faces.append([])
            edges.append([])
            tags.append(llPrefix + 'centroids')

        if withAdjacency:
            vertices.append(padHor(self.sites))
            faces.append([])
            edges.append(self.cellAdjacencyEdges)
            tags.append(llPrefix + 'adjacency')

        if withBBs:
            bbVerts = []
            bbEdges = []
            for sIdx in self.sIdxs:
                bbVerts.append(np.vstack([self.cellBBs[sIdx], self.cellBBcenters[sIdx]]))
                bbEdges.append(np.int32([[0, 1], [1, 2], [2, 3], [3, 0]]) + sIdx * 5)

            vertices.append(padHor(np.vstack(bbVerts)))
            faces.append([])
            edges.append(np.vstack(bbEdges))
            tags.append(llPrefix + 'BBs')

        writeObjFile(resDir + llPrefix + fileName, vertices, faces, edges, subTags=tags)

    def exportToSvg(self, fileName='infinityVoronoi.svg', size=512, withSites=True):
        theDom = minidom.parseString('<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"%d\" width=\"%d\"></svg>'%(size, size))
        for sIdx in self.sIdxs:
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
            for sIdx in self.sIdxs:
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


if __name__ == '__main__':

    n = 10
    e = 1.0
    sites = generateJitteredGridPoints(n, 2, e)

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
        # run the relaxation (156 iterations) and save the demo image as SVG
        cObj.lloydRelax(0.001)
        cObj.exportToSvg()