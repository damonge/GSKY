from gsky_theory import *

def main():
    Nz=[]
    for zm,zw in [(0.5,0.1),(0.8,0.2),(1.1,0.25)]:
        zarr=np.linspace(zm-3*zw,zm+3*zw,1000)
        Nzarr = np.exp(-(zm-zarr)**2/(2*zw**2))
        Nz.append((zarr,Nzarr))
    Nt=len(Nz)
    t=GSKY_Theory(Nz)
    larr=np.linspace(100,2000,20)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))

    for count in range(2):

        if (count==0):
            style='-'
            lf=lambda x:x
        else:
            style='--'
            lf=lambda x:None
            t.set_cosmology(ccl.Cosmology(Omega_c=0.3, Omega_b=0.045, h=0.67, sigma8=0.83, n_s=0.96))
            t.set_params({'mmin':11.5})
                    
        
        plt.subplot(2,4,1); plt.gca().set_prop_cycle(None)
        for i in range(Nt):
            for j in range(i,Nt):
                plt.plot (larr,t.getCls('gg',larr,i,j),style,label='GG %i x %i'%(i,j))


        plt.subplot(2,4,2); plt.gca().set_prop_cycle(None)
        for i in range(Nt):
            for j in range(Nt):
                plt.plot (larr,t.getCls('gs',larr,i,j),style,label=lf('GS %i x %i'%(i,j)))
            
        plt.subplot(2,4,3); plt.gca().set_prop_cycle(None)
        for i in range(Nt):
            for j in range(i,Nt):
                plt.plot (larr,t.getCls('ss',larr,i,j),style,label=lf('SS %i x %i'%(i,j)))

        plt.subplot(2,4,4); plt.gca().set_prop_cycle(None)
        for i in range(Nt):
            plt.plot (larr,t.getCls('gk',larr,i),style,label=lf('GK %i'%i))

        plt.subplot(2,4,5); plt.gca().set_prop_cycle(None)
        for i in range(Nt):
            plt.plot (larr,t.getCls('sk',larr,i),style,label=lf('SK %i'%i))

        plt.subplot(2,4,6); plt.gca().set_prop_cycle(None)
        for i in range(Nt):
            plt.plot (larr,t.getCls('gy',larr,i),style,label=lf('GY %i'%i))

        plt.subplot(2,4,7); plt.gca().set_prop_cycle(None)
        for i in range(Nt):
            plt.plot (larr,t.getCls('sy',larr,i),style,label=lf('SY %i'%i))

        plt.subplot(2,4,8); plt.gca().set_prop_cycle(None)
        plt.plot (larr,t.getCls('kk',larr),style,label=lf('KK'))

    for c in range(1,9):
        plt.subplot(2,4,c)
        plt.xlabel('$\ell$')
        plt.ylabel('$C_\ell$')
        plt.legend(fontsize=6)
        plt.semilogy();
        plt.tight_layout(); 

    plt.show()
            
    

    


    
if __name__=="__main__":
    main()
