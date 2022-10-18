from ChebyshevObject3D import *


if __name__== '__main__':

    outputPrefix = 'demo3D_'

    e = 1.0
    sites = np.random.rand(3**3, 3) * 2 * e - e


    # random sites axis aligned and randomly oriented
    aniFun = None
    for tag, oriFun in zip(['axisAligned', 'random'], [None, lambda xyzs: np.random.rand(len(xyzs), 3) * np.pi - np.pi/2]):
        cObj = ChebyshevObject3D(sites, oriFun, aniFun, e, tag = tag)
        cObj.computeDiagram()
        cObj.plot(withCentroids=False, withInitCells=False, withBBs=False)
        cObj.exportToObj(outputPrefix + tag + '.obj')


    # provoke cells of higher genus on tailored input
    # with axis aligned and rotated sites
    sites = np.float32([[-1,0,0],[0,0,0],[1,0,0]]) * 0.25
    def aniFun(xyzs): return np.tile([[1,2,1],[2,2,2],[1,1,2]], [1,2])

    for m, oriFun in zip(['AxisAligned', 'Rotated'], [None, lambda xyzs: np.float32([[0.25,0,0],[0,0,0],[0.25,0,0]]) * np.pi]):
        tag = 'genus'+m
        cObj = ChebyshevObject3D(sites, oriFun, aniFun, e, withMP = False, tag = tag)
        cObj.computeDiagram()
        cObj.exportToObj(outputPrefix + tag + '.obj')


    # wavy orientation field and anisotropic scaling
    # load relaxed CVT sites from file
    def oriFun(xyzs): return np.transpose(np.vstack([np.zeros((2,len(xyzs))), np.pi/2 - (1+np.cos(np.pi * xyzs[:,0])) * np.pi/20]))
    def aniFun(xyzs): return np.tile(((2-np.abs(xyzs[:,0])) * [[0],[1],[0]]).T + [1,0,1], [1,2])
    for m in '2i':
        tag = 'wavyL'+m
        sites = np.load('data/sitesL%s.npz'%m)['sites']
        cObj = ChebyshevObject3D(sites, oriFun, aniFun, e, tag = tag)
        cObj.computeDiagram()
        cObj.exportToObj(outputPrefix + tag + ".obj")