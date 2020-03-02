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

    plt.subplot(2,4,1)
    for i in range(Nt):
        for j in range(i,Nt):
            plt.plot (larr,t.getCls('gg',larr,i,j),label='GG %i x %i'%(i,j))
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()
    
    plt.subplot(2,4,2)
    for i in range(Nt):
        for j in range(Nt):
            plt.plot (larr,t.getCls('gs',larr,i,j),label='GS %i x %i'%(i,j))
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()
    
    plt.subplot(2,4,3)
    for i in range(Nt):
        for j in range(i,Nt):
            plt.plot (larr,t.getCls('ss',larr,i,j),label='SS %i x %i'%(i,j))
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()

    plt.subplot(2,4,4)
    for i in range(Nt):
        plt.plot (larr,t.getCls('gk',larr,i),label='GK %i'%i)
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()

    plt.subplot(2,4,5)
    for i in range(Nt):
        plt.plot (larr,t.getCls('sk',larr,i),label='SK %i'%i)
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()

    plt.subplot(2,4,6)
    for i in range(Nt):
        plt.plot (larr,t.getCls('gy',larr,i),label='GY %i'%i)
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()

    plt.subplot(2,4,7)
    for i in range(Nt):
        plt.plot (larr,t.getCls('sy',larr,i),label='SY %i'%i)
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()


    plt.subplot(2,4,8)
    plt.plot (larr,t.getCls('kk',larr),label='KK')
    plt.xlabel('$\ell$')
    plt.ylabel('$C_\ell$')
    plt.tight_layout(); plt.semilogy(); plt.legend()

    plt.show()
            
    

    


    
if __name__=="__main__":
    main()
