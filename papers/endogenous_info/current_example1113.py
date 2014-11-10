from numpy import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from mpl_toolkits.mplot3d import Axes3D
from openopt import NLP
from openopt import GLP
#from FuncDesigner import *
from scipy.optimize import *

#This program can generate the values for table 1 in the paper.
#Note that  the "#" sign comments out everything on the same line following the "#"

##We use the following shorthand notation
###ef is effort (e in the paper)
###efh is used for high effort (e^h in the paper)
###gam is a parameter in the cost function
###al is the share of high risk types in the population
###th is the probability of a high risk type to have an accident
###tl is the probability of a low risk type to have an accident
###ul"i" is the utility a contract "i" gives in the case of an accident
###ub"i" is the utility a contract "i" gives in the case of no accident
###ub0 is the no accident utility without insurance
###ul0 is the accident utility without insurance


##effort cost function
def cost(ef, gam):
    return gam*ef**4

##derivative cost function
def costp(ef,gam):
	return gam*4*ef**3

##perceived accident risk after getting a high signal
def betah(ef,al,th,tl):
    return ef*(1-al)*(th-tl)+al*th+(1-al)*tl

##derivative betah
def betahp(ef,al,th,tl):
	return (1-al)*(th-tl)

##perceived accident probability when getting a low signal
def betal(ef,al,th,tl):
    return al*th+(1-al)*tl-ef*al*(th-tl)

##derivative betal
def betalp(ef,al,th,tl):
	return -al*(th-tl)

## define utility function
def util(x):
	return sqrt(x)

##define the inverse utility function
def uinv(u):
	return u**2

####SEPARATING/EXCLUSION
##for the separating/exclusion case binding constraints are solved for the utility levels giving
def ub1(ef,efh,al,gam,ub0,ul0,th,tl):
	return ((cost(ef,gam)-cost(efh,gam))/(1-al)) +ub0-betal(efh,al,th,tl)*costp(efh,gam)/(al*betahp(efh,al,th,tl))-(betah(ef,al,th,tl)-betal(ef,al,th,tl))*costp(ef,gam)/(al*betahp(ef,al,th,tl))+cost(ef,gam)/al

def ul1(ef,efh,al,gam,ub0,ul0,th,tl):
	return ((cost(ef,gam)-cost(efh,gam))/(1-al)) +ul0+(1-betal(efh,al,th,tl))*costp(efh,gam)/(al*betahp(efh,al,th,tl))-(betah(ef,al,th,tl)-betal(ef,al,th,tl))*costp(ef,gam)/(al*betahp(ef,al,th,tl))+cost(ef,gam)/al

def ub2(ef,efh,al,gam,ub0,ul0,th,tl):
	return ((cost(ef,gam)-cost(efh,gam))/(1-al)) +ub0-betal(efh,al,th,tl)*costp(efh,gam)/(al*betahp(efh,al,th,tl))+betal(ef,al,th,tl)*costp(ef,gam)/(al*betahp(ef,al,th,tl))

def ul2(ef,efh,al,gam,ub0,ul0,th,tl):
	return ((cost(ef,gam)-cost(efh,gam))/(1-al)) +ul0+(1-betal(efh,al,th,tl))*costp(efh,gam)/(al*betahp(efh,al,th,tl))-(1-betal(ef,al,th,tl))*costp(ef,gam)/(al*betahp(ef,al,th,tl))

##profits in a separating menu as function of ef and efh
def profitsseparating(ef,efh,al,gam,w,D,th,tl):
	ub0 = util(w)
	ul0 = util(w-D)
	if efh>=ef:
		return w+al*(-betah(ef,al,th,tl)*(D+uinv(ul1(ef,efh,al,gam,ub0,ul0,th,tl)))-(1-betah(ef,al,th,tl))*uinv(ub1(ef,efh,al,gam,ub0,ul0,th,tl)))+(1-al)*(-betal(ef,al,th,tl)*(D+uinv(ul2(ef,efh,al,gam,ub0,ul0,th,tl)))-(1-betal(ef,al,th,tl))*uinv(ub2(ef,efh,al,gam,ub0,ul0,th,tl)))
	else:
		#print 'Warning: Input error ef>efh'
		return 0


###GRID-MAXIMIZATION
def gridmax(al,gam,w,D,th,tl,steps):
	grid = linspace(0.000,1.0,steps)
	pisep = 0
	for n in grid:
		gridh = linspace(n,1,steps-n*steps)
		for nh in gridh:
			pinowsep = profitsseparating(n,nh,al,gam,w,D,th,tl) 
			if pinowsep >pisep:
				pisep = pinowsep
				solutioneffortsep = [n,nh]
	x0 = [solutioneffortsep[0],solutioneffortsep[1]]
	lb = [0,0]
	ub = [1,1]
	A= [1,-1]
	b=[0]
	f=lambda x: -profitsseparating(x[0],x[1],al,gam,w,D,th,tl)#note that the functions below search for a minimum, hence the "-"
	p=NLP(f,x0,lb=lb,ub=ub,A=A,b=b,contol = 1e-8,gtol = 1e-10,ftol = 1e-12)
	solver='ralg'
	r=p.solve(solver)
	#the "2 program" simply assumes ef=0 and maximizes over efh only; the result is then compared to the other program above. This is done because there is a local maximum at ef=0 (see paper)
	f2=lambda x: -profitsseparating(0,x[0],al,gam,w,D,th,tl)
	lb2=[0]
	ulb2=[1]
	p2=NLP(f2,solutioneffortsep[1],lb=lb2,ub=ulb2,contol = 1e-8,gtol = 1e-10,ftol = 1e-12)
	r2=p2.solve(solver)
	if r.ff<r2.ff:
		print 'solver result with gamma=',gam,', alpha=',al,', w=',w,' and D=',D,', th=',th,', tl=',tl,' the effort levels are: ', r.xf
		ref=r.xf[0]
		refh=r.xf[1]
		piff=r.ff
	else:
		print 'solver result with gamma=',gam,', alpha=',al,', w=',w,' and D=',D,', th=',th,', tl=',tl,' the effort levels are : 0',r2.xf
		ref=0
		refh=r2.xf[0]
		piff=r2.ff
	print ref,refh
	print 'ub1 is ', ub1(ref,refh, al,gam,util(w),util(w-D),th,tl), '; ul1 is ',ul1(ref,refh, al,gam,util(w),util(w-D),th,tl)
	print 'ub2 is ', ub2(ref,refh, al,gam,util(w),util(w-D),th,tl), '; ul2 is ',ul2(ref,refh, al,gam,util(w),util(w-D),th,tl)
	euff=al*(betah(ref,al,th,tl)*ul1(ref,refh, al,gam,util(w),util(w-D),th,tl)+(1-betah(ref,al,th,tl))*ub1(ref,refh, al,gam,util(w),util(w-D),th,tl))+(1-al)*(betal(ref,al,th,tl)*ul2(ref,refh, al,gam,util(w),util(w-D),th,tl)+(1-betal(ref,al,th,tl))*ub2(ref,refh, al,gam,util(w),util(w-D),th,tl))-cost(ref,gam)
	print 'expected utility under this contract is ', euff
	print 'expected solver profits are ',-piff
	return [-piff,euff]#this return is used for creating the graph


###This prints the values in the table of the paper###
for ga in [0.05,0.1,0.2,0.5,0.7,1.0,1.3,1.5]:#[0.1,0.4,0.7,1.0,1.3,1.7,2.0,2.5,3.0]:
	gridmax(0.7,ga,4,3,0.35,0.2,500)


###########################################################
################Plot profit and expected utility as function of gamma###########
###########################################################
euplot=[]
piplot=[]
gamplot=[]

###This does the calculations for figure 2 in the paper
#for ga in linspace(0.05,1.3,26):
#	[a,b]=gridmax(0.7,ga,4,3,0.35,0.2,500)
#	piplot.append(a)
#	euplot.append(b)
#	gamplot.append(ga)
#
#print piplot,euplot,gamplot

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()

host.set_xlim(0.1, 1.3)
host.set_ylim(0.14, 0.17)

host.set_xlabel("$\\gamma$")
host.set_ylabel("Profits")
par1.set_ylabel("Utility")

p1, = host.plot(gamplot, piplot, 'b--',label="Profits")
p2, = par1.plot(gamplot, euplot,'g-', label="Utility")

par1.set_ylim(1.694, 1.71)


host.legend(loc=4)#this sets the legend into the lower right

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())

plt.draw()
plt.show()

plt.savefig('figure2.eps')
plt.savefig('figure2.pdf')


##########################################################################
#############Plots of profits as function of effort: Not used or needed in the paper####################
##########################################################################
###Plotting the objective function
def profitsplot(ef,efh,al,gam,w,D,th,tl):
	ub0 = util(w)
	ul0 = util(w-D)
	return w+al*(-betah(ef,al,th,tl)*(D+uinv(ul1(ef,efh,al,gam,ub0,ul0,th,tl)))-(1-betah(ef,al,th,tl))*uinv(ub1(ef,efh,al,gam,ub0,ul0,th,tl)))+(1-al)*(-betal(ef,al,th,tl)*(D+uinv(ul2(ef,efh,al,gam,ub0,ul0,th,tl)))-(1-betal(ef,al,th,tl))*uinv(ub2(ef,efh,al,gam,ub0,ul0,th,tl)))



#the following plot function removes the points where ef>efh from the plot; this allows to use profitsplot
#furthermore this function has the variables maxef and maxefh which allows to specifically zoom in around e=0 (maxef is the maximum effort level plotted)
def piplotrest(al,gam,w,D,th,tl,steps,maxef,maxefh):
	gridef = linspace(0.0,maxef,steps)
	gridefh = linspace(0.0,maxefh,steps)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	X=gridef
	Y=gridefh
	count = linspace(0,len(gridef)-1,len(gridef))
	X,Y=meshgrid(X,Y)
	Z=profitsplot(X,Y,al,gam,w,D,th,tl)
	for j in count:
		count2 = linspace(j+1,len(gridefh)-1,len(gridefh)-j-1)
		zvalue = profitsplot(0,0,al,gam,w,D,th,tl)
		for k in count2:
			X[j][k]=0
			Y[j][k]=0
			Z[j][k]=zvalue
	ax.set_xlabel('e')
	ax.set_ylabel('eh')
	ax.set_zlabel('profit')
	ax.plot_wireframe(X,Y,Z)#, rstride=10, cstride=10)
	#print X,Y
	#print Z
	plt.show()



##2D plot of profit as function of ef for a given efh
def plotpiofef(al,gam,w,D,th,tl,steps,efh):
	gridef = linspace(0.0,efh,steps)
	fig = plt.figure()
	y = profitsplot(gridef,efh,al,gam,w,D,th,tl) 
	plt.plot(gridef,y)
	plt.show()

##2D plot of profit as function of efh for a given ef
def plotpiofefh(al,gam,w,D,th,tl,steps,ef, maxefh):
	gridefh = linspace(ef,maxefh,steps)
	fig = plt.figure()
	y = profitsplot(ef,gridefh,al,gam,w,D,th,tl) 
	plt.plot(gridefh,y)
	plt.show()


