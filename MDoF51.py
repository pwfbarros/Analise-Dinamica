#BIBLIOTECAS
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#pd.options.display.float_format = '{:,.5f}'.format
import ROM51 as rom

#Modulo Oscilador
class oscilador:
    #Funcao Constructor
    def __init__(self, NumEQ, nstep, T, M, C, K1, F, NumF, V0, U0, K2):     
        self.NumEQ = NumEQ
        self.NumF = NumF
        self.T = T
        self.nstep = nstep         
        self.dt = self.T/self.nstep
        self.M = M
        self.C = C
        self.K1 = K1
        self.K2 = K2
        self.F = F
        self.V0 = V0
        self.U0 = U0
        self.I = np.identity(NumEQ)
        self.t = np.arange(self.nstep)*self.dt      
        #print("\nParametros do sistema:\nGDL = {0}\nTempo = {1:.2}s\ndt = {2}s " .format(self.NumEQ, self.T, self.dt))   
        #print("\nMatriz M:\n",self.M,"\n\nMatriz C:\n",self.C,"\n\nMatriz K:\n",self.K ,"\n\nVetor F:\n",self.F,"\n\nVetor V0:\n",self.V0)
    #Fim da Funcao Constructor
            

    #Função 3: Transforma vetores em unidimensional para plotar com linhas."
    def plotar (self, fig, dof = 1, color = None, ret = 0): 
        x = np.zeros(shape=(self.nstep))
        y = np.zeros(shape=(self.nstep))
        
        fig.suptitle('Resultado para o GDL '+ str(dof), fontsize=18, fontweight='bold')
        fig.subplots_adjust(hspace=0.5) 
        
        for i in range(self.nstep):
            x[i] = self.t[i]
            y[i] = self.U[dof-1,i]
        
        ax = fig.add_subplot(311)
        if color == None:        
            ax.plot(x[ret:], y[ret:], 'b-', linewidth=1.0)
        else:
            ax.plot(x[ret:], y[ret:], color+'-.', linewidth=1.0)
        ax.set_title('Deslocamento em função do tempo')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Deslocamento (m)')
        ax.grid(True)        
        
        for i in range(self.nstep):
            x[i] = self.t[i]
            y[i] = self.V[dof-1,i]
        
        ax = fig.add_subplot(312)
        if color == None:         
            ax.plot(x[ret:], y[ret:], 'g-', linewidth=1.0)
        else:
            ax.plot(x[ret:], y[ret:], color+'-.', linewidth=1.0)
        ax.set_title('Velocidade em função do tempo')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Velocidade (m/s)')
        ax.grid(True)           
        
        for i in range(self.nstep):
            x[i] = self.t[i]
            y[i] = self.A[dof-1,i]
        
        ax = fig.add_subplot(313)
        if color == None:         
            ax.plot(x[ret:], y[ret:], 'r-', linewidth=1.0)
        else:
            ax.plot(x[ret:], y[ret:], color+'-.', linewidth=1.0)
        ax.set_title('Aceleração em função do tempo')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Aceleração (m/s²)')
        ax.grid(True) 
        
        fig.subplots_adjust(top=0.85)
        return
    #Fim da Funcao 3


    #Função 9: Runge-Kutta nao linear2
    def runge_kutta_nl(self, yi):
        self.Ft = np.zeros(shape=(self.NumEQ,self.nstep))
        self.V = np.zeros(shape=(self.NumEQ,self.nstep))
        self.U = np.zeros(shape=(self.NumEQ,self.nstep))
        self.A = np.zeros(shape=(self.NumEQ,self.nstep))        
        
        self.Fmax = np.array(self.F[0:self.NumEQ,:])
        self.wf = np.array(self.F[self.NumEQ,:])
        for j in range(self.nstep):
            self.Ft[0,j] = self.Fmax*math.sin(self.wf*self.t[j])         

        def F(x,y,j):
            F = np.zeros(2)
            fnl = -rom.f2(yi+np.array([0.,y[0],0.]))#np.array([0.,self.K[0,0]*y[0],0.])#self.K[0,0]*y[0]
            #print(fnl[1] , -rom.f2(yi+np.array([0.,y[0],0.]))[1])
            F[0] = y[1]
            F[1] = (self.Ft[:,j]-self.C[0,0]*y[1]-fnl[1])/self.M[0,0]
            return F    
        
        def run_kut4_nl(F,x,y,h,j):
            self.RK0 = h*F(x,y,j)
            self.RK1 = h*F(x + h/2.0, y + self.RK0/2.0,j)
            self.RK2 = h*F(x + h/2.0, y + self.RK1/2.0,j)
            self.RK3 = h*F(x + h, y + self.RK2,j)
            return (self.RK0 + 2.0*self.RK1 + 2.0*self.RK2 + self.RK3)/6.0
        
        x = 0.0 # Start of integration; tempo
        #xStop = self.T # End of integration
        y = np.array([U0[0], V0[0]]) # Initial values of {y}
        h = self.dt # Step size        
        
        X = []
        Y = []
        A = []

        for j in range(self.nstep):
            y = y + run_kut4_nl(F,x,y,h,j)
            x = x + h
            a = F(x,y,j)[1]
            
            X.append(x)
            Y.append(y)
            A.append(a)
        
        X = np.array(X)
        Y = np.array(Y)
        A = np.array(A)

        self.U[0,:] = Y[:,0]
        self.V[0,:] = Y[:,1]
        self.A[0,:] = A
        
        self.Umax = np.amax(self.U[0,int(self.nstep*3/4):self.nstep])
        self.D = self.Umax/(self.Fmax[0,0]/K[0,0])
        print ("\n\nSolucao por Runge-Kutta NL:\nUmax = {0:.3} m\nD = {1}".format(self.Umax, self.D))        

          
    
#Função 10: Bathe nao-linear
    def bathe_nl(self, yi):
        self.V = np.zeros(shape=(self.NumEQ,self.nstep))
        self.U = np.zeros(shape=(self.NumEQ,self.nstep))
        self.A = np.zeros(shape=(self.NumEQ,self.nstep))        
        
        self.Fmax = np.array(self.F[0:self.NumEQ,:])
        self.wf = np.array(self.F[self.NumEQ,:])
        
        def Kb(X, dx = 1e-4):
            return (F(X+dx) - F(X))/dx     
        
        def F(X):
            fnl = -rom.f2(yi+np.array([0.,X,0.]))
            return fnl[1]
                       
        j=0    
        F(0)
        self.V[0,0] = self.V0
        self.U[0,0] = self.U0
        self.A[0,0] = (self.Fmax*math.sin(self.wf*self.t[0]) - self.C*self.V[0,0] - F(self.U[0,0]))/self.M

        
        for j in range(self.nstep-1):

            self.un = self.U[0,j]
            self.du = 10.
            while abs(self.du) > 1e-3:          
                self.kn = (16./self.dt**2)*self.M + 4./self.dt*self.C + Kb(self.un)
                self.rn = self.Fmax*math.sin(self.wf*(self.t[j] + self.dt/2.)) - F(self.un) - self.M*(16./self.dt**2*(self.un - self.U[0,j]) - 8./self.dt*self.V[0,j] - self.A[0,j]) - self.C*(4./self.dt*(self.un - self.U[0,j]) - self.V[0,j])               
                self.du = self.rn/self.kn
                self.un = self.un + self.du
            
            self.vn = (self.un - self.U[0,j])*(4.0/self.dt) - self.V[0,j]
            self.an = (self.vn - self.V[0,j])*(4.0/self.dt) - self.A[0,j]
        

            self.um = self.un          
            self.du = 10.
            while abs(self.du) > 1e-3:
                self.km = (9./self.dt**2)*self.M + 3./self.dt*self.C + Kb(self.um)
                self.rm = self.Fmax*math.sin(self.wf*self.t[j+1]) - F(self.um) - self.M*(9/self.dt**2*self.um - 12/self.dt**2*self.un + 3./self.dt**2*self.U[0,j] - 4/self.dt*self.vn + self.V[0,j]/self.dt) - self.C*(3./self.dt*self.um - 4./self.dt*self.un + self.U[0,j]/self.dt)
                self.du = self.rm/self.km            
                self.um = self.um + self.du               
            
            self.vm = self.U[0,j]/self.dt - 4./self.dt*self.un + 3./self.dt*self.um
            self.am = self.V[0,j]/self.dt - 4./self.dt*self.vn + 3./self.dt*self.vm
        
            self.U[0,j+1] = self.um
            self.V[0,j+1] = self.vm
            self.A[0,j+1] = self.am
     
        self.Umax = np.amax(self.U[0,int(self.nstep*3/4):self.nstep])
        self.D = self.Umax/(self.Fmax/K[0,0])
        print ("\n\nSolucao por Bathe NL:\nUmax = {0:.3} m\nD = {1}".format(self.Umax, self.D))          
        

def integra(x0, cic, temporal = None, retro_t = 0, poincare = None, espectro = None, ret = -600):            
    
    print('ret = {0}'.format(ret))
    o1.bathe_nl(np.array(x0))
    #o1.runge_kutta_nl(np.array(x0))
    obj = o1.U
    avg = np.average(obj[0,ret:])
    sp = np.fft.fft(obj[0,ret:]-avg)
    freq = np.fft.fftfreq(len(obj[0,ret:]), d=o1.dt)

    dw = 1./(-ret)
    data = pd.DataFrame({'rad/s':2*math.pi*freq,'Imaginario':sp.imag,'Real':sp.real,
                         'abs(A)':np.abs(sp),'2*dw*abs(A)':2*dw*np.abs(sp),'phase(A)':np.angle(sp),
                         'U0': o1.U0[0], 'V0': o1.V0[0]})
    
    if temporal != None:    
        o1.plotar(temporal, ret = retro_t)
    
    if poincare != None:    
        ax3 = poincare.add_subplot(111)
        ax3.plot(o1.U[0,ret:] , o1.V[0,ret:], 'b', linewidth=1.0)
        mci = cic[int(len(cic)*-ret/o1.nstep):]
        ax3.plot(o1.U[0,ret:][mci] , o1.V[0,ret:][mci], 'ro', linewidth=1.0)
        ax3.set_title('Trajetória no Espaço de Fase e Seção de Poincaré')
        ax3.set_xlabel('Posição (m)')
        ax3.set_ylabel('Velocidade (m/s)')
        ax3.grid(True)
        
        """
        ax3.set_ylim([0.06,-0.06])
        xx = 1.7-rom.n.dy
        ax3.plot([xx,xx] , [0.6,-0.6], 'k--', linewidth=0.5)
        """
       
    if espectro != None:
        ax4 = espectro.add_subplot(111)
        msk = (data.loc[:,'rad/s']>0.) & (data.loc[:,'rad/s']<3.5*o1.F[1,0])
    
        x = np.array(data.loc[msk,'rad/s'])
        y = np.array(data.loc[msk,'2*dw*abs(A)'])
        bwidth = 0.9*float(data.loc[1,'rad/s'])
        ax4.bar(x,y,bwidth,align='center')
    
        ax4.set_title('Espectro de Amplitudes')
        ax4.set_xlabel('Frequência (rad/s)')
        ax4.set_ylabel('Amplitude (m)')
        ax4.grid(True)
        #ax4.set_xticks(data.loc[msk,'rad/s'])
       
    #print('\n')
    #print(data[data['rad/s']>=0.].sort_values(by='abs(A)',ascending=False)[:])

    
    return (data[data['rad/s']==0].sort_values(by='abs(A)',ascending=False).iloc[0],
            data[data['rad/s']>0.].sort_values(by='abs(A)',ascending=False).iloc[0],
            data[data['rad/s']>0.].sort_values(by='abs(A)',ascending=False).iloc[1],
            data[data['rad/s']>0.].sort_values(by='abs(A)',ascending=False).iloc[2],
            avg, o1.Umax)

def mapa(nx, ny, lim_x, lim_y, save = None):
    dx,dy = lim_x/(nx-1) , lim_y/(ny-1)

    x, y = np.mgrid[0:nx,0:ny]
    x = (x-(nx-1)/2)*lim_x/((nx-1)/2)
    y = (y-(ny-1)/2)*lim_y/((ny-1)/2)
    z = np.zeros(shape=(nx,ny))

    x0 = np.array(rom.pos_f)
    data = pd.DataFrame()

    cont = 0
    for j in range(ny):
        for i in range(nx):
            o1.U0[0] = x[i,j]
            o1.V0[0] = y[i,j]  
            re = integra(x0,cic)
            cont = cont + 1
            print(cont)       
                
            re[0]['freq'] = int(0)
            re[0]['2*dw*abs(A)'] = re[4]
            data = data.append(re[0])
            
            re[1]['freq'] = int(1)
            data = data.append(re[1])
            z[i,j] = re[5]
            
            re[2]['freq'] = int(2)       
            data = data.append(re[2])
            
            re[3]['freq'] = int(3)
            data = data.append(re[3])
    
    z_min, z_max = 0., np.abs(z).max()

    plt.figure(figsize=(8,6),facecolor='white')
    plt.suptitle('Amplitude da Frequência Dominante com n = '+ str(n), fontsize=12, fontweight='bold')
    plt.subplots_adjust(hspace=2.5)

    fig, ax = plt.subplots(1, 1)

    nxp, nyp = nx+1, ny+1

    xp, yp = np.mgrid[0:nxp,0:nyp]
    xp = (xp-(nxp-1)/2)*(lim_x+dx)/((nxp-1)/2)
    yp = (yp-(nyp-1)/2)*(lim_y+dy)/((nyp-1)/2)

    plt.pcolor(xp, yp, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    # set the limits of the plot to the limits of the data
    plt.axis([xp.min(), xp.max(), yp.min(), yp.max()])
    #plt.axis('equal')
    plt.colorbar()
    plt.xlabel('Posição inicial (m)')
    plt.ylabel('Velocidade inicial (m/s)')

    c = np.ones_like(x)
    ax.pcolor(xp, yp, c, facecolor='none', edgecolor='k')

    
    if save != None :   
        plt.savefig(save+'.pdf')

        writer = pd.ExcelWriter(save+'.xlsx')
        data.to_excel(writer,'Sheet1')
        pd.DataFrame(z).to_excel(writer,'Sheet2')
        writer.save()
    


#________________________________________________________________________________________________________

#Início do código
import timeit
start = timeit.default_timer()

ndof = 1
nforcas = 1

M = np.zeros(shape=(ndof,ndof))
C = np.zeros(shape=(ndof,ndof))
K1 = np.zeros(shape=(ndof,ndof))
K2 = np.zeros(shape=(ndof,ndof))

F = np.zeros(shape=(ndof+1,nforcas))
V0 = np.zeros(shape=(ndof))
U0 = np.zeros(shape=(ndof))

M = M + np.array([235000*1e3])


K1 = K1 + np.array([12353280])
K2 = K2 + np.array([54898470])
K = (4.0*K1*K2/(math.sqrt(K1)+math.sqrt(K2))**2)

C = 2.0*M*math.sqrt(K/M)*0.02

w = math.sqrt(K/M)
n = 2.

F[0,0] = K[0,0]/5.4
F[ndof,0] =  n*w

V0 = V0 + np.array([0.])
U0 = U0 + np.array([0.]) 

"""
Frequencias naturais
 U0      w (rad/s)
-0.500   0.3130
-0.300   0.3220
-0.250   0.3270
-0.200   0.3335
-0.150   0.3454
-0.100   0.3679 (*)
-0.075   0.3891 (*)
-0.050   0.4255 (*)
-0.025   0.4828 (*)
-0.010   0.4829
-0.001   0.4829


Instabilidades 

n = 3. / 3 pontos / csi = 0.02
U0 =  -0.4   
V0 =  -0.4
mapa com Bathe, div = 20, T = 400*dt*div

n = 2.7 / 5 pontos / csi = 0.02 (Period Doubling Bifurcation 2x, 4x)
U0 = 0.0 0.2   
V0 = 0.0 0.2

n = 2. / 2 pontos / csi = 0.1 # Esta é a razão por que não se pode utilizar uma análise linear (Incluir comparação na dissertação)
U0 =  0.  
V0 =  0.
"""

div = 20
dt = (2*math.pi/w)/(n*div)
T = 400*dt*div
nstep = round(T/dt)

step_ciclo = div
#n_ciclos = int(round(nstep/step_ciclo)-1)
cic = np.arange(0,nstep)%step_ciclo==0 #numero de passos para completar um ciclo do carregamento harmonico

o1 = oscilador(ndof, nstep, T, M, C, K1, F, nforcas, V0, U0, K2)
x0 = np.array(rom.pos_f)

poincare = plt.figure()
espectro = plt.figure()


temporal = plt.figure(figsize=(20,12),facecolor='white')
retro_t = int(-nstep/16)
ret = int(-nstep/2)


data = pd.DataFrame()
re = integra(x0, cic, temporal=temporal, retro_t=retro_t, poincare=poincare, espectro=espectro, ret=ret)
re[0]['freq'] = int(0)
re[0]['2*dw*abs(A)'] = re[4]
data = data.append(re[0])
re[1]['freq'] = int(1)
data = data.append(re[1])
re[2]['freq'] = int(2)       
data = data.append(re[2])
re[3]['freq'] = int(3)
data = data.append(re[3])




"""
nx, ny = 51, 51
lim_x = 0.5
lim_y = 0.5
save = 'Umax_n'+str(n)
mapa(nx,ny,lim_x,lim_y,save=save)
"""


"""
#períodos naturais
n = 1
V0[0] = 0
F[0,0] = K[0,0]*0
C = 2.0*M*math.sqrt(K/M)*0
U0tab = [-0.1, -0.075, -0.05, -0.025]
wntab = [0.3679, 0.3891, 0.4255, 0.4828]
pic = plt.figure()
for i in range(len(U0tab)):
    w = wntab[i]
    F[ndof,0] =  w
    U0[0] = U0tab[i]
    div = 40
    dt = (2*math.pi/w)/(n*div)
    T = 20*dt*div
    nstep = round(T/dt)
    step_ciclo = div
    cic = np.arange(0,nstep)%step_ciclo==0 
    o1 = oscilador(ndof, nstep, T, M, C, K1, F, nforcas, V0, U0, K2)
    x0 = np.array(rom.pos_f)
    retro_t = int(-nstep)
    ret = int(-nstep/2)
    data = pd.DataFrame()
    re = integra(x0, cic, temporal=temporal, retro_t=retro_t, poincare=pic, espectro=espectro, ret=ret)
    
    #ax = pic.axes[0]
    #ax.set_title("")
"""

stop = timeit.default_timer()
print('\n\nExec time:',stop - start) 