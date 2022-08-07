from ChebyshevObject2D import *

def generateJitteredGridPoints(n, e):
    rng = np.linspace(-e, e, n, endpoint=False) + e / n
    xyGrid = np.vstack(np.dstack(np.meshgrid(rng, rng)))
    return xyGrid + randomJitter(n**2, 2, e / n)    

if __name__== '__main__':

    outputPrefix = 'demo2D_'

    e = 1.0
    sites = generateJitteredGridPoints(9, e)

    # all sites axis aligned and unformly scaled
    tag = 'axisAligned'
    oriFun = None
    aniFun = None

    cObj = ChebyshevObject2D(sites, oriFun, aniFun, e, tag = tag)
    cObj.computeDiagram()

    cObj.plot(withBBs = False, fileName = outputPrefix + tag + '.pdf')
    cObj.exportToObj(outputPrefix + tag + '.obj')


    # wavy orientation field and anisotropic scaling
    tag = 'wave'
    oriFun = lambda xys: np.pi / 2 - (1 + np.cos(np.pi * xys[:,0] / e)) / 2
    aniFun = lambda xys: 2 - np.tile([np.ones(len(xys)), np.abs(xys[:,0])], [2,1]).T

    cObj = ChebyshevObject2D(sites, oriFun, aniFun, e, tag = tag)
    cObj.computeDiagram()

    cObj.plot(fileName = outputPrefix + tag + '.pdf')
    cObj.exportToObj(outputPrefix + tag + '.obj')


    # 10 LLoyd relaxation steps with circular orientation field
    tag = 'relaxedCircular'
    oriFun = lambda xys: np.arctan2(normVec(xys)[:,1], normVec(xys)[:,0])
    aniFun = lambda xys: [0.5] * 4 + norm(xys).reshape(-1,1)

    cObj = ChebyshevObject2D(sites, oriFun, aniFun, e, tag = tag)
    cObj.lloydRelax(10)

    cObj.plot(fileName = outputPrefix + tag + '.pdf')
    cObj.exportToObj(outputPrefix + tag + '.obj')
