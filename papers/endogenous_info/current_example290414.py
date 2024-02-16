from numpy import *
import matplotlib.pyplot as plt
from FuncDesigner import *
from scipy.optimize import *



###v is share of low risk types
###th is the probability of a high risk type to have an accident
###tl is the probability of a low risk type to have an accident
###c is cost of effort
###w is initial wealth
###d is damage
###uba is high coverage utility with accident
###ubn is high coverage utility without accident
###ula is low coverage utility with accident
###uln is low coverage utility without accident

###ATTENTION: the code uses directly the fact that h is quadratic and would have to be changed substantially if another (non-square root) utility function was used!!!

uba, ubn, ula, uln = oovars('uba', 'ubn', 'ula', 'uln')

#utility function
def u(x):
	return sqrt(x)

#derivative of the utility function
def up(x):
	return 1/(2*sqrt(x))

#inverse utility function
def h(x):
	return x*x

#derivative of h
def hp(x):
	return 2*x

#####INFORMATION ACQUISITION#####

#expected utility under separating
def EUs(uba,ubn,ula,uln,w,d,th,tl,v,c):
	return v*((1-tl)*uln+tl*ula)+(1-v)*((1-th)*ubn+th*uba)-c

#IR-low
def IRl(uba,ubn,ula,uln,w,d,th,tl,v,c):
	return (1-tl)*(uln-u(w))+tl*(ula-u(w-d))

#IG-low
def IGl(uba,ubn,ula,uln,w,d,th,tl,v,c):
	return EUs(uba,ubn,ula,uln,w,d,th,tl,v,c)-v*((1-tl)*uln+tl*ula)-(1-v)*((1-th)*uln+th*ula)

##both types active##
#equation that is satisfied in the optimal contract, see proposition 3
def eqopts(uba,ubn,ula,uln,w,d,th,tl,v,c):
	return v*tl*(1-tl)*(hp(uln)-hp(ula))-(1-v)*(th-tl)*hp(uba)

#high coverage contract gives full coverage
def fullcovh(uba,ubn,ula,uln,w,d,th,tl,v,c):
	return uba-ubn

#returns the optimal separating profit when both types are insured; returns -1000 if the equations yield an infeasible solution (in this case exclusion is better than insuring both types)
def pisepa(w,d,th,tl,v,c):
	f=[IRl(uba,ubn,ula,uln,w,d,th,tl,v,c),IGl(uba,ubn,ula,uln,w,d,th,tl,v,c),fullcovh(uba,ubn,ula,uln,w,d,th,tl,v,c),eqopts(uba,ubn,ula,uln,w,d,th,tl,v,c)]
	linSys=sle(f)
	r = linSys.solve()
	UBAS, UBNS, ULAS, ULNS = r(uba,ubn,ula,uln)
	if ULNS>u(w) or ULAS<u(w-d):
		return [-1000]
	else:
		return [w-d*(v*tl+(1-v)*th)-v*((1-tl)*h(ULNS)+tl*h(ULAS))-(1-v)*h(UBNS),UBAS,UBNS,ULAS,ULNS]


##only high demand types active##
#profits directly taken from proposition 3
def pisepone(w,d,th,tl,v,c):
	return (1-v)*(w-th*d-h((1-th)*u(w)+th*u(w-d)+c/(1-v)))


##Combine##
#returns maximal profit from inducing inforamtion gathering as well as the expected utility of the agent
def pisep(w,d,th,tl,v,c):
	sep = pisepa(w,d,th,tl,v,c)
	pirealsep = sep[0]
	piexclusion = pisepone(w,d,th,tl,v,c)
	if pirealsep==-1000:
		return [piexclusion,EUs((1-th)*u(w)+th*u(w-d)+c/(1-v),(1-th)*u(w)+th*u(w-d)+c/(1-v),u(w-d),u(w),w,d,th,tl,v,c)]
	elif pirealsep>piexclusion:
		ubag = sep[1]
		ubng = sep[2]
		ulag = sep[3]
		ulng = sep[4]
		#print ubag, ubng, ulag, ulng
		return [pirealsep,EUs(ubag,ubng,ulag,ulng,w,d,th,tl,v,c)]
	else:
		return [piexclusion,EUs((1-th)*u(w)+th*u(w-d)+c/(1-v),(1-th)*u(w)+th*u(w-d)+c/(1-v),u(w-d),u(w),w,d,th,tl,v,c)]


#####NO INFORMATION ACQUISITION#####

#expected utility in no information acquisition
def EUpool(ua,un,w,d,th,tl,v,c):
	return (v*(1-tl)+(1-v)*(1-th))*un+(v*tl+(1-v)*th)*ua

#IG-high
def IGh(ua,un,w,d,th,tl,v,c):
	return (v*(1-tl)+(1-v)*(1-th))*un+(v*tl+(1-v)*th)*ua-v*(tl*u(w-d)+(1-tl)*u(w))-(1-v)*((1-th)*un+th*ua)+c

#optimality condition when c<c' (see proposition 4, equation 27 in the paper); ATTENTION: This uses the fact that h is quadratice, i.e. this has to be changed if a non-square-root utility function is used
def focnoinfo(ua,un,w,d,th,tl,v,c):
	return (v*tl*(1-tl)+(1-v)*th*(1-tl))*ua-(v*tl*(1-tl)+(1-v)*tl*(1-th))*un

#c'' as defined in the paper
def cdp(w,d,th,tl,v):
	return v*(1-v)*(th-tl)*(u(w)-u(w-d))

#c' as defined in text; if c' does not exist 0 is returned
def cp(w,d,th,tl,v):
	if up(w)/up(w-d)>(v*tl*(1-tl)+(1-v)*tl*(1-th))/(v*tl*(1-tl)+(1-v)*th*(1-tl)):
		return 0
	else:
		phipa = -(v*tl+(1-v)*th)/(v*(1-tl)+(1-v)*(1-th))
		phiph = -tl/(1-tl)
		te=v*tl+(1-v)*th
		return (v*(1-v)*(th-tl)*(phiph*u(w)-phipa*u(w-d)))/((1-te)*phipa+te*phiph)

#this determines which case we are in, i.e. whether c is below/above c' or c'' and assigns the corresponding profit; it returns an array whose first component is expected profit and the second component is expected utility
def pipool(w,d,th,tl,v,c):
	te=v*tl+(1-v)*th
	k=c/(v*(1-v)*(th-tl))
	if c<cp(w,d,th,tl,v):
		f=[IGh(uba,ubn,w,d,th,tl,v,c),focnoinfo(uba,ubn,w,d,th,tl,v,c)]
		linSys=sle(f)
		r = linSys.solve()
		UA, UN = r(uba, ubn)
		#print UA, UN
		return [w-te*d-h(UN)*(v*(1-tl)+(1-v)*(1-th))-h(UA)*(v*tl+(1-v)*th),EUpool(UA,UN,w,d,th,tl,v,c)]
	elif c<cdp(w,d,th,tl,v):
		return [w-te*d-h(u(w)-te*k)*(v*(1-tl)+(1-v)*(1-th))-h(u(w-d)+(1-te)*k)*(v*tl+(1-v)*th),EUpool(u(w-d)+(1-te)*k,u(w)-te*k,w,d,th,tl,v,c)]
	else:
		return [w-te*d-h(v*((1-tl)*u(w)+tl*u(w-d))+(1-v)*((1-th)*u(w)+th*u(w-d))),EUpool(v*((1-tl)*u(w)+tl*u(w-d))+(1-v)*((1-th)*u(w)+th*u(w-d)),v*((1-tl)*u(w)+tl*u(w-d))+(1-v)*((1-th)*u(w)+th*u(w-d)),w,d,th,tl,v,c)]


#####CALCULATING c*, checking expected utility and aggregate profits#####

def cstar(w,d,th,tl,v):
	return fsolve(lambda x: pipool(w,d,th,tl,v,x)[0]-pisep(w,d,th,tl,v,x)[0], 0.001)[0]
	
#difference between left and right limit of expected utility at c^*
def eudif(w,d,th,tl,v):
	print 'difference of expected utility of info gathering and no gathering at c* is ', pisep(w,d,th,tl,v,cstar(w,d,th,tl,v))[1]- pipool(w,d,th,tl,v,cstar(w,d,th,tl,v))[1]

#profits under optimal contract
def profits(w,d,th,tl,v,c):
	poolprof = pipool(w,d,th,tl,v,c)[0]
	sepprof = pisep(w,d,th,tl,v,c)[0]
	return maximum(poolprof,sepprof)



##this calculates the cost cutoff values in our numerical example
#eudif(100,51,0.7,0.4,0.8)
print cstar(100,51,0.7,0.4,0.8), pisep(100,51,0.7,0.4,0.8,cstar(100,51,0.7,0.4,0.8)), pipool(100,51,0.7,0.4,0.8,cstar(100,51,0.7,0.4,0.8))
print cp(100,51,0.7,0.4,0.8)
print cdp(100,51,0.7,0.4,0.8)

## this plugs in cstar for the values and can be used to get the equilibrium contracts: just  remove the comment symbol "#"
#pisep(100,51,0.7,0.4,0.8,0.00585415523138) 
#pipool(100,51,0.7,0.4,0.8,0.00585415523138)


#####PLOT#####
def plotpi(w,d,th,tl,v):
	grid = linspace(0.0,0.2,100)
	vecprof = vectorize(lambda x: profits(w,d,th,tl,v,x))
	vecprofsep = vectorize(lambda x: pisep(w,d,th,tl,v,x)[0])
	vecprofpool = vectorize(lambda x: pipool(w,d,th,tl,v,x)[0])
	fig = plt.figure()
	y = vecprof(grid) 
	ysep = vecprofsep(grid)
	ypool = vecprofpool(grid)
	plt.plot(grid,y,'b',grid,ysep,'r--',grid,ypool,'g--')
	plt.xlabel(r"c")
	plt.ylabel(r"profits")
	plt.legend(('profits','$\\pi_{x=1}^*$','$\\pi_{x=0}^*$'),'lower left', fancybox=True)
	plt.savefig('profits.eps')
	plt.savefig('profits.pdf')
	plt.show()


def ploteu(w,d,th,tl,v):
	cs=cstar(w,d,th,tl,v)
	grid1 = linspace(0.0,cs,50)
	grid2 = linspace(cs,0.15,50)
	veceu1 = vectorize(lambda x: pisep(w,d,th,tl,v,x)[1])
	veceu2 = vectorize(lambda x: pipool(w,d,th,tl,v,x)[1])
	fig = plt.figure()
	y1= veceu1(grid1)
	y2= veceu2(grid2)
	plt.plot(grid1,y1,'b',grid2,y2,'b',linewidth=3.0)
	plt.xlabel(r"c")
	plt.ylabel(r"expected utility")
	plt.savefig('utility.eps')
	plt.savefig('utility.pdf')
	plt.show()

## this creates the plot of the expected utility that we use in the paper (figure 2)
ploteu(100,51,0.7,0.4,0.8)
