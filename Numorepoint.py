import matplotlib
import matplotlib.pyplot as plt
import numpy as np

markers=['o','s','^','>','<','v','o','s','^','>','<','v']
#colors=[(164/256,5/256,69/256),(244/256,111/256,68/256),(253/256,217/256,133/256),(233/256,245/256,161/256),(127/256,203/256,164/256),(75/256,101/256,175/256)]
#colors=[(2/256,124/256,165/256),(32/256,128/256,165/256),(58/256,133/256,165/256),(93/256,138/256,165/256),(141/256,146/256,165/256),(168/256,151/256,165/256),(192/256,157/256,165/256),(220/256,160/256,165/256),(250/256,165/256,165/256)]
colors=[(68/256,117/256,122/256),(69/256,42/256,61/256),(212/256,76/256,60/256),(221/256,108/256,76/256),(229/256,133/256,93/256),(238/256,213/256,183/256)]
#colors=[(38/256,70/256,83/256),(40/256,114/256,113/256),(42/256,157/256,140/256),(138/256,176/256,125/256),(233/256,196/256,107/256),(243/256,162/256,97/256),(230/256,111/256,81/256)]
colors=[(0.74,0.14,0.18),(0.93,0.32,0.23),(0.96,0.42,0.2),(0.96,0.58,0.19),(0.99,0.85,0.52),(0.5,0.8,0.95),(0.38,0.75,0.91),(0.2,0.6,0.85),(0.16,0.56,0.8),(0.11,0.44,0.71)]
colors=[(0.74,0.14,0.18),(0.93,0.32,0.23),(0.96,0.42,0.2),(0.99,0.85,0.52),(0.5,0.8,0.95),(0.38,0.75,0.91),(0.16,0.56,0.8),(0.11,0.44,0.71)]
colors=colors[::-1]

colors=[(0.74,0.14,0.18),(0.93,0.32,0.23),(0.96,0.42,0.2),(0.96,0.58,0.19),(0.91,0.77,0.42),(0.99,0.85,0.52),(0.6,0.81,0.99),(0.5,0.8,0.95),(0.38,0.75,0.91),(0.2,0.6,0.85),(0.16,0.56,0.8),(0.11,0.44,0.71)]
colors=[(209/256,146/256,192/256),(105/256,141/256,201/256),(154/256,187/256,80/256),(249/256,185/256,119/256),(220/256,33/256,65/256),(200/256,33/256,50/256),(105/256,141/256,201/256)]
colors=colors[::-1]
labels=['$3.3 \\times 10^{-1}$','$1.8 \\times 10^{-1}$','$9.5 \\times 10^{-2}$','$9.5 \\times 10^{-2}$','$6.6 \\times 10^{-2}$','$2.5 \\times 10^{-1}$','$1.4 \\times 10^{-1}$']
markers=['o','s','^','D','o','v','>','<','^','>','<','v']
#-- Plot setting
if True:
    matplotlib.rcParams['axes.linewidth']       = 1
    matplotlib.rcParams['xtick.major.size']     = 12
    matplotlib.rcParams['xtick.major.width']    = 1
    matplotlib.rcParams['xtick.minor.size']     = 6
    matplotlib.rcParams['xtick.minor.width']    = 1
    matplotlib.rcParams['ytick.major.size']     = 10
    matplotlib.rcParams['ytick.major.width']    = 1
    matplotlib.rcParams['ytick.minor.size']     = 6
    matplotlib.rcParams['ytick.minor.width']    = 1
    matplotlib.rcParams['ytick.direction']     = 'in'
    matplotlib.rcParams['xtick.direction']    = 'in'

    matplotlib.rcParams['xtick.major.pad']      = 10
    matplotlib.rcParams['ytick.major.pad']      = 10

    matplotlib.rcParams['mathtext.default']     = 'regular'
#-- End
# 0 0.01 0.1 0.3 0.5 1 2 3 5 10 20 30 50 100 200 300
xx =[0.01,0.1,0.3,0.5,1,2,3,5,10,20,30,50,100]



Nu = np.loadtxt('./data_figure/Nu_lamda.txt',delimiter=',')

y = [1,6,2,7,4,5]
Nu = np.array(Nu)
plt.figure()
fig,ax = plt.subplots(figsize=(6.5, 5))
calpha=1.0

# background
x1,x2 = 2e-1,2e2
y1,y2 = 0.8,2.0
# plt.xlim([x1,x2])
# plt.ylim([y1,y2])
c1 = 'red'
c2 = 'skyblue'
# plt.fill([x1,10,10,x1],[y1,y1,y2,y2],c1,alpha=0.1,zorder=1)
# plt.fill([10,x2,x2,10],[y1,y1,y2,y2],c2,alpha=0.1,zorder=1)
# y2 = 0.85
# plt.fill([x1,10,10,x1],[y1,y1,y2,y2],c1,alpha=0.2,zorder=1)
# plt.fill([10,x2,x2,10],[y1,y1,y2,y2],c2,alpha=0.2,zorder=2)
# y1 = 1.95
# y2 = 2.0
# plt.fill([x1,10,10,x1],[y1,y1,y2,y2],c1,alpha=0.2,zorder=1)
# plt.fill([10,x2,x2,10],[y1,y1,y2,y2],c2,alpha=0.2,zorder=2)

for j in y:
    Nu[j-1,:] = Nu[j-1,:]/11.8
    plt.plot(xx[2:],Nu[j-1,4:],'o-',lw=1.5,color=colors[y.index(j)],markersize=1,alpha=1,zorder=j)
    plt.scatter(xx[2:],Nu[j-1,4:],marker=markers[j],color='w',edgecolors=colors[y.index(j)],s=120,lw=1,label=labels[j-1],zorder=j+1)

#3d
X = [0,0.3,0.5,1,2,3,5,10]
X2 = [0*0.095/0.22,0.3*0.095/0.22,1*0.095/0.22,2*0.095/0.22,3*0.095/0.22,5*0.095/0.22,10*0.095/0.22]
Y = [16.0/17.45,15.93/17.45,16.5/17.45,17.45/17.45,16.98/17.45,17.55/17.45,19.5/17.45]
Y = np.array([15.356133,16.254356,17.103601,17.380328,16.814019,17.675821,20.701983])
Y = Y/15.75
# plt.scatter(X2,Y,marker=markers[3],color='Grey',s=150,lw=2,label='$\lambda$=0.095(3D)')
plt.plot(X[1:],Y[:],'o-',lw=1.5,color='black',markersize=1,alpha=1,zorder=9)
plt.scatter(X[1:],Y[:],marker=markers[3],color='w',edgecolors='black',s=120,lw=1,label='$9.5 \\times 10^{-2}$(3D)',zorder=10)

# plt.figtext(0.175, 0.72, '$\lambda$:', fontsize = 14)
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9),ncol=2,frameon=False,numpoints=2,fontsize=14,columnspacing=0.8,handletextpad=0.8)

omega1 = [1,1.3,1.5,2,2.3,2.5,3]
omega2 = [2,2.3,2.5,3,4,5]
Nu_5 = [13.422248,13.388432,13.438927,14.217587,14.275126,14.141721,12.927133]
Nu_7 = [13.072983,13.467945,13.495196,13.356326,13.080288,12.895518,13.570751]
Nu_10 = [13.045875,13.213497,13.522576,14.218349,14.109835,13.968208,13.570751]
Nu_13 = [12.876914,13.246557,13.531897,13.895650,13.885093,13.968936,14.299412]
Nu_20 = [14.393014,14.816235,15.196819,15.815532,15.819998,14.653386]
Nu_24 = [14.069829,14.397479,14.633276,15.411641,15.676245,14.939222] # x 162745
Nu_30 = [13.169232,13.201717,13.211527,13.527497,13.525540,13.482834]
plt.plot(omega1,np.array(Nu_5)/11.8,'o-',lw=1.5,color=colors[y.index(1)],markersize=1,alpha=1,zorder=1)
plt.scatter(omega1,np.array(Nu_5)/11.8,marker=markers[1],color='w',edgecolors=colors[y.index(1)],s=120,lw=1,label=labels[0],zorder=2)
plt.plot(omega1,np.array(Nu_7)/11.8,'o-',lw=1.5,color=colors[y.index(6)],markersize=1,alpha=1,zorder=6)
plt.scatter(omega1,np.array(Nu_7)/11.8,marker=markers[6],color='w',edgecolors=colors[y.index(6)],s=120,lw=1,label=labels[5],zorder=7)
# plt.plot(x1[:],y1[:],'--',lw=1.5,color='grey',alpha=1)
# plt.figtext(-0.03, 0.87, '(a)', fontsize = 22)
plt.figtext(0.16, 0.8, 'Distance between cilia: $\lambda$', fontsize = 14)
# plt.figtext(0.26, 0.145, 'Regime I', fontsize = 14,color=c1,alpha=0.6)
# plt.figtext(0.65, 0.145, 'Regime II', fontsize = 14,color=c2,alpha=1)
plt.axhline(y=1,c='grey',ls='--',lw=1,zorder=0)
# plt.axvline(x=10,c='grey',ls='--',lw=1,zorder=0)

plt.xlim([2e-1,2e2])
plt.ylim([0.8,2.0])
plt.ylabel(u'$Nu(\omega)$/$Nu_{RB}$',fontsize=24)
# plt.xlabel(u'$\lambda$/$l_{cor}$',fontsize=24)
plt.xlabel(u'$\omega$',fontsize=24)
for label in ax.get_xticklabels():
    label.set_fontsize(18)
for label in ax.get_yticklabels():
    label.set_fontsize(18)

plt.semilogx()
# plt.semilogy()
# plt.savefig('./Nu_1.pdf', bbox_inches='tight', dpi=256)

plt.show()

