from ChebyshevObject2D import *


if __name__== '__main__':

    outputPrefix = 'demo2D_'

    e = 1.0
    sites = generateJitteredGridPoints(5, 2, e)


    # examples with different but uniform anisotropic site weights
    oriFun = None
    tags = ['lambda1', 'Lambda2121', 'Lambda1234']
    aniFuns = [None, lambda xys: np.tile([2,1,2,1],[len(xys),1]), lambda xys: np.tile([1,2,3,4],[len(xys),1])]
    for tag, aniFun in zip(tags, aniFuns):
        cObj = ChebyshevObject2D(sites, oriFun, aniFun, e, tag = tag)
        cObj.computeDiagram()
        cObj.plot(withBBs = False, fileName = outputPrefix + tag + '.pdf')
        cObj.exportToObj(outputPrefix + tag + '.obj')


    # more sites for relaxations
    sites = generateJitteredGridPoints(9, 2, e)

    # wavy orientation field and anisotropic scaling -> runs 139 iterations
    # circular orientation field and position dependent scaling -> runs 134 iterations
    oriFuns = [lambda xys: np.pi/2 - (1 + np.cos(np.pi * xys[:,0] / e)) * np.pi/20, lambda xys: np.arctan2(normVec(xys)[:,1], normVec(xys)[:,0])]
    aniFuns = [lambda xys: 2 - np.tile([np.ones(len(xys)), np.abs(xys[:,0])], [2,1]).T, lambda xys: [0.5] * 4 + norm(xys).reshape(-1,1)]
    for m, oriFun, aniFun in zip(['Wavy', 'Circular'], oriFuns, aniFuns):
        tag = 'relaxed'+m
        cObj = ChebyshevObject2D(sites, oriFun, aniFun, e, tag = tag)
        cObj.lloydRelax(0.001)

        cObj.plot(fileName = outputPrefix + tag + '.pdf')
        cObj.exportToObj(outputPrefix + tag + '.obj')