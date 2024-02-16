using Roots
using JuMP
using Clp
using Base.Cartesian
using NLopt
using Ipopt
using Calculus

######################################
########OPTIMAL MECHANISM#############
######################################

function linProgCons(a,b,w,g)
    """maximize a*y s.t.: b*y>=0 and 0<=y[i]<=1 where a and b are Vectors/Arrays/Lists of the same length subject to monotonicity constraint"""
    n = length(a) #number of elements of y
    kb = length(w) #number of buyer signals
    ks = length(g) #number of seller signals
    myModel = Model(with_optimizer(Clp.Optimizer,LogLevel=0,PrimalTolerance=1e-8,DualTolerance=1e-8));
    @variable(myModel, 0.0<=y[i=1:n]<=1.0);
    @constraint(myModel,con, sum(b[i]*y[i] for i=1:n)>=0.0);
    @objective(myModel, Max, sum(a[i]*y[i] for i=1:n));
    for j in 1:ks-1
        @constraint(myModel, sum(w[i]*y[i+(j-1)*kb] for i=1:kb)-sum(w[i]*y[i+j*kb] for i=1:kb)>=0.0)#monotonicity seller
    end
    for j in 1:kb-1
        @constraint(myModel, sum(g[i]*y[j+1+(i-1)*kb] for i=1:ks)-sum(g[i]*y[j+(i-1)*kb] for i=1:ks)>=0.0)#monotonicity buyer
    end
    JuMP.optimize!(myModel)
    #println(termination_status(myModel),primal_status(myModel),dual_status(myModel))
    return objective_value(myModel), [value(y[i]) for i=1:n]
end

function WMaxMech(v,c,w,g)
    """welfare maximizing mechanism if buyer (seller) has valuation (cost) vector v (c) with probabilities w (g); v and c are ordered from lowest to highest"""
    W = similar(w)
    cumsum!(W,w) #cdf of pdf w
    G = similar(g)
    cumsum!(G,g) #cdf for pdf g
    n = length(w)
    m = length(g)
    a = zeros(Float64,n,m) #coefficients of y in objective
    b = zeros(Float64,n,m) #coefficients of y in BB constraint
    for i in 1:n-1
        a[i,1] = (v[i]-c[1])*g[1]*w[i]
        b[i,1] = g[1]*(v[i]*w[i]+(v[i]-v[i+1])*(1-W[i]) )-w[i]*c[1]*g[1]
        for j in 2:m
            a[i,j] = (v[i]-c[j])*g[j]*w[i]
            b[i,j] = g[j]*(v[i]*w[i]+(v[i]-v[i+1])*(1-W[i]) )-w[i]*(c[j]*g[j]+G[j-1]*(c[j]-c[j-1]))
        end
    end
    a[n,1] = (v[n]-c[1])*g[1]*w[n]
    b[n,1] = g[1]*(v[n]*w[n] )-w[n]*c[1]*g[1]
    for j in 2:m
        a[n,j] = (v[n]-c[j])*g[j]*w[n]
        b[n,j] = g[j]*(v[n]*w[n] )-w[n]*(c[j]*g[j]+G[j-1]*(c[j]-c[j-1]))
    end
    welfare,y = linProgCons(a,b,w,g)
    return welfare, reshape(y,(n,m))
end

########################################################################
#########OPTIMAL SIGNAL STRUCTURE FOR UNIFORM TYPE DISTRIBUTION#########
########################################################################

 """myLoop(A::Array{T,N},B::Array{S,M};arguments=())


walks through a grid of N cutoffs with gridsize 1/M for two players and calls welfareCutoffs at each gridpoint
"""
@generated function myLoop(A::Array{T,N},B::Array{S,M};arguments=()) where {T,N,S,M}
    quote
        step = 1/$M  # step size          
        dummy = $N+1
        eval(:($(Symbol("j_$dummy")) =0.0)) # creates the variable j_{N+1} and sets it to 0
        @nloops $N j j->(j_{j+1}+step:step:1.0-j*step+step/100) begin #"+step/100 is there to avoid a missing step due to numerical imprecisions
            if arguments == ()
                myLoop(A,B,arguments=@ntuple $N j) #go through the grid for the other player
            else
                @ncall $N welfareCutoffs arguments... j #calls welfareCutoffs with arguments (j_1,j_2,...,j_N)
            end
        end
    end
end


"""
wMaxInfoUni(n,stepsPer1)

Calculates the welfare maximzing info structure with n+1 seller and n+1 buyer signals if the true distribution is uniform on [0,1] using a gridsize of 1/stepPer1
"""
function wMaxInfoUni(n,stepsPer1)
    global welfareOpt, cutOpt
    welfareOpt = 0.0 #initialization
    myLoop(Array{Float64}(undef,[1 for i=1:n]...),Array{Float64}(undef,[1 for i=1:stepsPer1]...)) #as values of arguments are not accessible in generated functions but types are, I use this trick to get the number of cutoffs and the stepsize into the loop
    return welfareOpt, cutOpt
end
        
"""
welfareCutoffs(x)

Computes welfare given a list of interior cutoffs where the first n are the buyer and the second n are the seller cutoffs assuming that true valuation and cost are uniformly distributed on [0,1]
"""
function welfareCutoffs(x...)
    global welfareOpt, cutOpt
    v,c,w,g = cutoffsToInfo(x)
    welfareNow = WMaxMech(v,c,w,g)[1] 
    if welfareNow>welfareOpt
        welfareOpt = welfareNow
        cutOpt = x
    end
end


"""
cutoffsToInfo(x)

Returns signal vectors and prob distribution given a cutoff vector x in which the first (second) n elements are interior cutoffs of the buyer (seller) assuming uniform distribution of both players' types on [0,1]
"""
function cutoffsToInfo(x)
    n = Int64(length(x)/2) # number of interior cut points per player
    bc = vcat(1.0,x[1:n]...,0.0) # buyer cutoffs
    sc = vcat(1.0,x[n+1:end]...,0.0) # seller cutoffs
    v = reverse([(bc[i]+bc[i+1])/2 for i in 1:n+1]) #reverse as x from loop is ordered from high to low values
    c = reverse([(sc[i]+sc[i+1])/2 for i in 1:n+1])
    w = reverse([(bc[i]-bc[i+1]) for i in 1:n+1])
    g = reverse([(sc[i]-sc[i+1]) for i in 1:n+1])
    return consolidateInfoStruc(v,c,w,g,n)
end

"""
consolidateInfoStruc(v,c,w,g,n;tol::Float64=1.0e-6)

deletes types that have zero probability
"""
function consolidateInfoStruc(v,c,w,g,n;tol::Float64=1.0e-6)
    vMask = [w[i]>tol for i in 1:n+1] #note that n is the number of /interior/ cutoffs per player
    cMask = [g[i]>tol for i in 1:n+1]
    return v[vMask],c[cMask],w[vMask],g[cMask]
end        


########use optimization algorithm NLopt##############

"""
mergeSignals(v,c,w,g,y)

merge two adjacent signals if they have the same y (as shown in the paper, this will improve welfare)
"""
function mergeSignals(v,c,w,g,y)
    n, m = size(y)
    i = 1
    flag = false #indicates whether signals were merged
    while i<n-1
        if y[i,:]==y[i+1,:]
            flag = true
            v[i] = (w[i]*v[i]+w[i+1]*v[i+1])/(w[i]+w[i+1])
            w[i] = w[i]+w[i+1]
            deleteat!(v,i+1)
            deleteat!(w,i+1)
            y=y[1:end.!=i+1 ,:]
            n = n-1
        else
            i=i+1
        end
    end
    i = 1
    while i<m-1
        if y[:,i]==y[:,i+1]
            flag = true
            c[i] = (g[i]*c[i]+g[i+1]*c[i+1])/(g[i]+g[i+1])
            g[i] = g[i]+g[i+1]
            deleteat!(c,i+1)
            deleteat!(g,i+1)
            y=y[:,1:end.!=i+1]
            m = m-1
        else
            i=i+1
        end
    end
    return v,c,w,g,y,flag
end

#only used for NLopt algorithm below
"""
welfareCutoffsArray(x::Vector)

Computes welfare given a list of interior cutoffs where the first n are the buyer and the second n are the seller cutoffs assuming that true valuation and cost are uniformly distributed on [0,1]
"""
function welfareCutoffsArray(x::Vector)
    
    v,c,w,g = cutoffsToInfo(x)
    welfare,y = WMaxMech(v,c,w,g)
    return welfare
end

welfareGradient = x->Calculus.gradient(welfareCutoffsArray,x)

"""
obj(x::Vector,grad::Vector)

objective function for NLopt with in place evaluation of gradient by finite differencing
"""
function obj(x::Vector,grad::Vector)
    if length(grad)>0
        grad[:] = welfareGradient(x)
    end
    return welfareCutoffsArray(x)
end

"""
monoConstraint(x::Vector,grad::Vector,i)

constraint for NLopt with in place evaluation of gradient: monotonicity of cutoffs
"""
function monoConstraint(x::Vector,grad::Vector,i)
    if length(grad) > 0
        grad[:] = zeros(Float64,length(grad))
        grad[i] = -1.0
        grad[i+1] = 1.0
    end
    return x[i+1]-x[i]
end

"""
wOptimization(n)

computes welfare maximizing information structure in a two step procedure: first evaluating the objective on a grid and second using a local maximization algorithm starting from the optimum in the first step
"""
function wOptimization(n)
    # find starting point by grid search
    nStep = max(min(Int64(ceil(n+1+10000^(1/n))),20),n+4)
    wStart, cutStart = wMaxInfoUni(n,nStep)
    cutStartArray = [i for i in cutStart] #[0.95,0.9,0.1,0.05]
    println(wStart," ",cutStartArray)
    # optimization with NLopt
    opt = Opt(:LD_MMA,2*n) # :LD_MMA is a local algorith using derivatives 
    lower_bounds!(opt, zeros(Float64,2*n))
    upper_bounds!(opt, ones(Float64,2*n))
    max_objective!(opt,obj)
    for i=1:n-1
        inequality_constraint!(opt,(x,g)->monoConstraint(x,g,i),0.0)
    end
    for i=n+1:2*n-1
        inequality_constraint!(opt,(x,g)->monoConstraint(x,g,i),0.0)
    end
    xtol_abs!(opt,1e-8)
    maxeval!(opt,5000)
    (maxW,maxCut,ret) = NLopt.optimize(opt,cutStartArray)
    infoOpt = cutoffsToInfo(maxCut)
    wOpt= WMaxMech(infoOpt...)
    return wOpt, infoOpt,ret,maxCut
end

for n in 1:6
    println("for ",n," interior cutoffs, the result is: ",wOptimization(n))
end

for n in 10:5:20
    println("for ",n," interior cutoffs, the result is: ",wOptimization(n))
end





