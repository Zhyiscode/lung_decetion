import matplotlib
import matplotlib.pyplot as plt
import numpy as np


colors=[(209/256,146/256,192/256),(105/256,141/256,201/256),(154/256,187/256,80/256),(249/256,185/256,119/256),(220/256,33/256,65/256),(200/256,33/256,50/256),(105/256,141/256,201/256)]
colors=colors[::-1]
labels=['$3.3 \\times 10^{-1}$','$1.8 \\times 10^{-1}$','$9.5 \\times 10^{-2}$','$8.0 \\times 10^{-2}$','$6.6 \\times 10^{-2}$','$2.5 \\times 10^{-1}$','$1.43 \\times 10^{-1}$']
markers=['o','s','^','D','o','v','>','<','^','>','<','v']
markers=['o','s','>','^','<','o','v','>','<','v']
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
Numax = np.loadtxt('./data_figure/Nu_max.txt',delimiter=',')
Numin = np.loadtxt('./data_figure/Nu_min.txt',delimiter=',')

y = [1,6,2,7,4,5]

Nu = np.array(Nu)
plt.figure()
fig,ax = plt.subplots(figsize=(6.5, 5))
calpha=1.0

x1,x2 = 2e-2,2e2
y1,y2 = 0.8,2.0

Numax = Numax/10
Numin = Numin/10
# print(Numax)

y = [1,6,2,7,4,5]

'''for k in [0,1,2,3,4,5]:
    Numax[k,:] = Numax[k,:]
    Numin[k,:] = Numin[k,:]'''
c = np.array([2,1.5,2.3,3,3.5,3.5])
# o_5
xx =[0.3,0.5,1,1.5,2,3,5,10,20,30,50,100]
Nu = [12.030749,12.365584,13.385814,13.404245,13.450890,12.755067,11.844562,12.272202,12.637338,13.553442,16.674192,19.512122]
Nu[:] = np.array(Nu[:])/13.4

plt.plot(xx[:]/c[0],Nu[:],'o-',lw=1.5,color=colors[0],markersize=1,alpha=1,zorder=0)
# plt.errorbar(xx[:]/c[0],Nu[:], yerr=[Numin[0,2:],Numin[0,2:]],ecolor=colors[0],elinewidth=0.1,marker=markers[1],mfc='white',mew=1,ms=10,alpha=1,capsize=6,capthick=3,linestyle="none")
plt.scatter(xx[:]/c[0],Nu[:],marker=markers[1],color='w',edgecolors=colors[0],s=120,lw=1,zorder=1)
# o_7
xx =[0.3,0.5,1,1.5,2,3,5,10,20,30,50,100]
Nu = [12.323374,11.931710,13.433850,13.495417,13.356326,12.551749,12.210199,11.900364,12.765715,14.766657,17.048669,20.417116]
Nu[:] = np.array(Nu[:])/13.49
plt.plot(xx[:]/c[1],Nu[:],'o-',lw=1.5,color=colors[1],markersize=1,alpha=1,zorder=1)
# plt.errorbar(xx[:]/c[0],Nu[:], yerr=[Numin[1,2:],Numax[1,2:]],ecolor=colors[1],elinewidth=0.1,marker=markers[2],mfc='white',mew=1,ms=10,alpha=1,capsize=6,capthick=3,linestyle="none")
plt.scatter(xx[:]/c[1],Nu[:],marker=markers[2],color='w',edgecolors=colors[1],s=120,lw=1,zorder=2)

# o_10
xx =[0.3,0.5,1,2.3,3,5,10,20,30,50,100]
Nu = [12.317759,11.819064,13.043112,14.051773,13.502620,12.718731,12.296604,13.208138,14.382249,17.503666,20.575385]
Nu[:] = np.array(Nu[:])/14.05
plt.plot(xx[:]/c[2],Nu[:],'o-',lw=1.5,color=colors[2],markersize=1,alpha=1,zorder=2)
# plt.errorbar(xx[:]/c[0],Nu[:], yerr=[Numin[0,3:],Numax[0,3:]],ecolor=colors[2],elinewidth=0.1,marker=markers[3],mfc='white',mew=1,ms=10,alpha=1,capsize=6,capthick=3,linestyle="none")
plt.scatter(xx[:]/c[2],Nu[:],marker=markers[3],color='w',edgecolors=colors[2],s=120,lw=1,zorder=3)

# o_13
xx =[0.3,0.5,1,2,3,5,10,20,30,50,100]
Nu = [12.545936,12.102554,12.876914,13.895650,14.299412,13.351070,12.164275,13.004302,14.905491,17.948762,20.210395]
Nu[:] = np.array(Nu[:])/14.3
plt.plot(xx[:]/c[3],Nu[:],'o-',lw=1.5,color=colors[3],markersize=1,alpha=1,zorder=3)
# plt.errorbar(xx[:]/c[0],Nu[:], yerr=[Numin[0,:],Numax[0,:]],ecolor=colors[0],elinewidth=0.1,marker=markers[1],mfc='white',mew=1,ms=10,alpha=1,capsize=6,capthick=3,linestyle="none")
plt.scatter(xx[:]/c[3],Nu[:],marker=markers[4],color='w',edgecolors=colors[3],s=120,lw=1,zorder=4)

# o_20
xx =[0.3,0.5,1,2,3.5,5,10,20,30,50,100]
Nu = [12.030805,11.691366,12.024584,13.979985,15.745014,14.570328,13.485268,13.775760,14.687578,16.738567,20.735521]
Nu[:] = np.array(Nu[:])/15.74
plt.plot(xx[:]/c[4],Nu[:],'o-',lw=1.5,color=colors[4],markersize=1,alpha=1,zorder=4)
# plt.errorbar(xx[:]/c[0],Nu[:], yerr=[Numin[0,:],Numax[0,:]],ecolor=colors[0],elinewidth=0.1,marker=markers[1],mfc='white',mew=1,ms=10,alpha=1,capsize=6,capthick=3,linestyle="none")
plt.scatter(xx[:]/c[4],Nu[:],marker=markers[5],color='w',edgecolors=colors[4],s=120,lw=1,zorder=5)

# o_30
xx =[0.3,0.5,1,2,3.5,5,10,20,30,50,100]
Nu = [11.550799,11.371238,11.404762,12.426692,13.548157,13.284517,13.038047,14.328426,16.351812,18.604461,20.898948]
Nu[:] = np.array(Nu[:])/13.548
plt.plot(xx[:]/c[5],Nu[:],'o-',lw=1.5,color=colors[5],markersize=1,alpha=1,zorder=5)
# plt.errorbar(xx[:]/c[0],Nu[:], yerr=[Numin[0,:],Numax[0,:]],ecolor=colors[0],elinewidth=0.1,marker=markers[1],mfc='white',mew=1,ms=10,alpha=1,capsize=6,capthick=3,linestyle="none")
plt.scatter(xx[:]/c[5],Nu[:],marker=markers[6],color='w',edgecolors=colors[5],s=120,lw=1,zorder=6)



# 3d
X = [0,0.3,0.5,1,1.5,2,3,5,10]
X = np.array(X)
Y = np.array([15.356133,16.254356,17.103601,17.308328,16.814019,17.675821,20.701983])
Y = np.array([15.83,15.2537,15.462,16.341,16.325,16.31,16.44,18.4])
Ymax = np.array([15.852,15.3945,15.464,16.346,16.309,16.325,16.54,18.82])
Ymin = np.array([15.825,15.37,15.461,16.336,16.331,16.298,16.35,18.43])
# Y = Y/17

plt.plot(X[1:]/1.5,Y[:]/16.341,'o-',lw=1.5,color='black',markersize=1,alpha=1,zorder=9)
plt.scatter(X[1:]/1.5,Y[:]/16.341,marker='D',color='w',edgecolors='black',s=120,lw=1,zorder=10)
# plt.errorbar(X[1:]/1.5,Y[:]/16.341, yerr=[Y-Ymin,Ymax-Y],ecolor='black',elinewidth=0.1,marker=markers[3],mfc='white',mew=1,ms=10,alpha=0.8,capsize=6,capthick=3)
# plt.figtext(0.175, 0.72, '$\lambda$:', fontsize = 14)
plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9),ncol=2,frameon=False,numpoints=2,fontsize=14,columnspacing=0.8,handletextpad=0.8)

# plt.axhline(y=1,c='grey',ls='--',lw=1,zorder=0)
#
plt.xlim([2e-2,2e2])
plt.ylim([0.6,1.8])
plt.ylabel(u'$Nu(\omega)$/$Nu(0)$',fontsize=24)

plt.xlabel(u'$\omega$',fontsize=24)
for label in ax.get_xticklabels():
    label.set_fontsize(18)
for label in ax.get_yticklabels():
    label.set_fontsize(18)

plt.semilogx()
# plt.semilogy()
plt.savefig('./Nu_n.pdf', bbox_inches='tight', dpi=256)

#plt.show()

