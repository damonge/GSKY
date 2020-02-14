import numpy as np

class ObsCond(object):
    def __init__(self,name,nx,ny,cutoff=-9999.) :
        """
        Observing condition object.
        :param name: name of the OC.
        :param data: list of OC values
        :param nx,ny: dimensionality of the output map
        :param cutoff: remove all data below the cutoff.
        """
        self.name=name
        self.cutoff=cutoff
        self.npix=nx*ny

        #Extra space at the end
        self.vmap=[[] for i in range(nx*ny)]
        self.wmap=[[] for i in range(nx*ny)]
        self.completed=False

    def add_frame(self,ipixs,val,weights):
        if self.completed:
            raise ValueError("I thought I was done!")

        if val>self.cutoff:
            for ip,w in zip(ipixs,weights):
                self.vmap[ip].append(val)
                self.wmap[ip].append(w)

    def complete_map(self):
        for ip,(v,w) in enumerate(zip(self.vmap,self.wmap)):
            self.vmap[ip]=np.array(v)
            self.wmap[ip]=np.array(w)

    def collapse_map_mean(self):
        map_out=np.zeros(self.npix)

        for ip,(v,w) in enumerate(zip(self.vmap,self.wmap)):
            wt=np.sum(w)
            if wt>0:
                map_out[ip]=np.sum(v*w)/wt
            else:
                map_out[ip]=-9999.
        return map_out

    def collapse_map_std(self):
        map_out=np.zeros(self.npix)

        for ip,(v,w) in enumerate(zip(self.vmap,self.wmap)):
            wt=np.sum(w)
            if wt>0:
                map_out[ip]=np.sum(v**2*w)/wt-(np.sum(v*w)/wt)**2
            else:
                map_out[ip]=-9999.
        return map_out

    def collapse_map_median(self):
        map_out=np.zeros(self.npix)

        for ip,(v,w) in enumerate(zip(self.vmap,self.wmap)):
            wt=np.sum(w)
            if wt>0:
                map_out[ip]=np.median(v)
            else:
                map_out[ip]=-9999.
        return map_out
