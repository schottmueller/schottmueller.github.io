using Plots

using Distributed
addprocs(11)

@everywhere using Roots
@everywhere using Optim

# parameter example
@everywhere struct Ex
	wb
	gu
	vu
	vb
	cu
	cb
end

"""
BB(gl,wh,p)

BB>=0 is budget balance constraint
"""
@everywhere BB(gl,wh,p)=gl*p.vb*(p.wb-wh)/(1-wh)+gl*p.vu*(1-p.wb)/(1-wh)+(1-gl)*wh*p.vb-wh*p.cu*(p.gu-gl)/(1-gl)-wh*p.cb*(1-p.gu)/(1-gl)-gl*(1-wh)*p.cu

"""
whBBRoot(gl,p)

returns the wh such that budget balance constraint holds with equality for a given gl and parameters p (uses root finding)
"""
@everywhere function whBBRoot(gl,p)
	return fzero(wh->BB(gl,wh,p),p.wb/2)
    end

"""
whBB(gl,p)

returns the wh such that budget balance constraint holds with equality for a given gl and parameters p (uses explicit solution)
"""
@everywhere function whBB(gl,p)
	denom = ((1-p.gu)*(p.cb-p.cu)/(1-gl)-(1-gl)*(p.vb-p.cu))
	innerterm = 1+gl*(p.vb-p.cu)/denom
	sol = 0.5*innerterm-sqrt(0.25*innerterm^2-(p.wb*gl*(p.vb-p.vu)+gl*(p.vu-p.cu))/denom)
	return sol                      
end

"""returns the gl such that budget balance holds with equality for a given wh and parameters p (uses root finding)"""
@everywhere  function glBBRoot(wh,p)
      return fzero(gl->BB(gl,wh,p),p.gu/2)
  end

@everywhere function glBB(wh,p)
      """returns the gl such that budget balance holds with equality for a given wh and parameters p (uses explicit solution)"""
      denom = (1-p.wb)*(p.vb-p.vu)/(1-wh)-(1-wh)*(p.vb-p.cu)
      innerterm = 1+wh*(p.vb-p.cu)/denom
      #sol1 = 0.5*innerterm+sqrt(0.25*innerterm^2-(wh*(p.vb-p.cb)+wh*p.gu*(p.cb-p.cu))/denom)
      sol2 = 0.5*innerterm-sqrt(0.25*innerterm^2-(wh*(p.vb-p.cb)+wh*p.gu*(p.cb-p.cu))/denom)
      return sol2
end

"""
welfare(gl,wh,p)

welfare if y(v_l,c_h)=0 and y()=1 for all other signals
"""
@everywhere function welfare(gl,wh,p)
      return p.vb*(wh*(1-gl)+gl*p.wb)+p.vu*gl*(1-p.wb)-p.cu*(gl*(1-wh)+wh*p.gu)-p.cb*wh*(1-p.gu)
end

"""
welfare(gl,p)

welfare if y(v_l,c_h)=0 and y()=1 for all other signals
"""
@everywhere function welfare(gl,p)
      welfare(gl,whBB(gl,p),p)
end

"""
Wfb(p)

computes first best welfare
"""
@everywhere function Wfb(p)
      return p.wb*p.vb+(1-p.wb)*p.gu*p.vu-p.gu*p.cu-(1-p.gu)*p.wb*p.cb
  end


""" 
optimum(p)

computes expected welfare maximum, returns optimal gl,wh,welfare
"""
@everywhere function optimum(p)
      lowestWH = whBB(p.gu,p)
      lowestGL = glBB(p.wb,p)
      opt = optimize(gl->(-1)*welfare(gl,p),lowestGL,p.gu)
      intW = (-1)*opt.minimum
      cornerW1 = welfare(lowestGL,p.wb,p)
      cornerW2 = welfare(p.gu,lowestWH,p)
      if intW > max(cornerW1,cornerW2)
	  return opt.minimizer, whBB(opt.minimizer,p), intW
      elseif cornerW1>cornerW2
	  return lowestGL,p.wb,cornerW1
      else
	  return p.gu,lowestWH,cornerW2
      end
  end


  function Loop(probstep,step,range)
      minWRatio = 1.0 # initialization
      pMinWRatio = Ex(0.0,0.0,1.0,3.0,0.0,2.0) #dummy initialization
      for wb in 0.1:probstep:0.9
	  for gu in 0.1:probstep:0.9
	      for cu in 0.0:step:range
		  for vu in cu+step:step:cu+range+step
		      for cb in vu+step:step:vu+range+step
			  for vb in cb+step:step:cb+range+step
			      p = Ex(wb,gu,vu,vb,cu,cb)
			      if BB(gu,wb,p)<-1e-6
				  opt = optimum(p)
				  if opt[1]<gu-1e-6 && opt[2]<wb-1e-6
				      println(wb," ",gu," ",vu," ",vb," ",cu," ",cb)
				  end
				  WRatio = opt[3]/Wfb(p)
				  if WRatio<minWRatio
				      minWRatio = WRatio
				      pMinWRatio = p
				  end
			      end
			  end
		      end
		  end
	      end
	  end
      end
      println("Done")
      return minWRatio, pMinWRatio
  end

function LoopParallel(probstep,step,range)    
    function inner(wb)
        minWRatio = 1.0 # initialization
        pMinWRatio = Ex(0.0,0.0,1.0,3.0,0.0,2.0) #dummy initialization
	for gu in 0.02:probstep:0.98
	    for cu in 0.0:step:range
	        for vu in cu+step:step:cu+range+step
		    for cb in vu+step:step:vu+range+step
		        for vb in cb+step:step:cb+range+step
			    p = Ex(wb,gu,vu,vb,cu,cb)
			    if BB(gu,wb,p)<-1e-6
			        opt = optimum(p)
				if opt[1]<gu-1e-6 && opt[2]<wb-1e-6
				    println(wb," ",gu," ",vu," ",vb," ",cu," ",cb)
				end
				WRatio = opt[3]/Wfb(p)
				if WRatio<minWRatio
				    minWRatio = WRatio
				    pMinWRatio = p
				end
			    end
                        end
		    end
	        end
	    end
        end
        return minWRatio, pMinWRatio
    end
    result = pmap(inner, collect(0.02:probstep:0.98))
    println("Done")
    return result
end

#output = @time LoopParallel(0.02,0.1,8.0)
#println(output[argmin([output[i][1] for i in 1:length(output)])])
#(0.9565824300248604, Ex(0.1, 0.1, 0.7999999999999999, 5.1, 0.7, 5.0)) # LoopParallel(0.05,0.1,5.0)

############################################
#####Convexity of W when moving along BB####
############################################

@everywhere using ForwardDiff
@everywhere function WprimeAlongBB(p)
    W(wh::Vector)=welfare(glBB(wh[1],p),wh[1],p)
    Wprime(wh)=ForwardDiff.gradient(W,[wh])[1]
    W2prime(wh)=ForwardDiff.hessian(W,[wh])[1]
    whl = whBB(p.gu,p)
    W2prime.(whl:(p.wb-whl)/25:p.wb)
    #return plot(whl:(p.wb-whl)/100:p.wb,Wprime.(whl:(p.wb-whl)/100:p.wb))
end


@everywhere function LoopW2prime(probstep,step,range)
     function inner(wb)
	  for gu in 0.1:probstep:0.9
	      for cu in 0.0:step:range
		  for vu in cu+step:step:cu+range+step
		      for cb in vu+step:step:vu+range+step
			  for vb in cb+step:step:cb+range+step
			      p = Ex(wb,gu,vu,vb,cu,cb)
			      if BB(gu,wb,p)<-1e-6
				  Deriv2 = WprimeAlongBB(p)
				  if !all(Deriv2.>=0.0)
				      println(wb," ",gu," ",vu," ",vb," ",cu," ",cb)
				  end
			      end
			  end
		      end
		  end
	      end
	  end
     end
     result = pmap(inner, collect(0.02:probstep:0.98))
      println("Done")
  end

#@time LoopW2prime(0.025,0.1,5.0) # 36443.641204 seconds (3.57 k allocations: 136.266 KiB)

##################################################
##########Welfare Ratio###########################
##################################################

"""
maxWelfare(p)

computes welfare if 1 or 2 signals are given per type and returns maximal welfare
"""
@everywhere function maxWelfare(p)
    W11 = max(0.0,p.wb*p.vb+(1-p.wb)*p.vu-p.gu*p.cu-(1-p.gu)*p.cb) #1 signal for buyer and 1 signal for seller
    W12 = (p.wb*p.vb+(1-p.wb)*p.vu-p.cu)*p.gu #pooling buyer, separating seller
    W21 = p.wb*(p.vb-p.gu*p.cu-(1-p.gu)*p.cb) #pooling seller, separating buyer
    W22 = max(welfare(p.gu,whBB(p.gu,p),p),welfare(glBB(p.wb,p),p.wb,p))
    return max(W11,W12,W21,W22)
end

"""
 loopGeneral(step,f,dummystart=1.0)

fixing vb to 1 loops through parameter grid with stepsizes "step" and evaluates f at every point
"""
@everywhere function loopGeneral(step,f,dummystart=1.0)
    function inner(wb)
        dummy = dummystart
        for gu in step:step:1-step
	    for cu in 0.0:step:1-3*step
		for vu in cu+step:step:1-2*step
		    for cb in vu+step:step:1-step
		        p = Ex(wb,gu,vu,1.0,cu,cb)
		        if BB(gu,wb,p)<-1e-6
			    dummy = f(p,dummy)
                        end
		    end
		end
	    end
	end
        return dummy
     end
    result = pmap(inner, collect(step:step:1-step))
    println("Done")
    return minimum(result)
end

@everywhere function Wratio(p)
    W2ndBest = maxWelfare(p)
    W1stBest = Wfb(p)
    return W2ndBest/W1stBest
end

@everywhere function eval(p,dummy)
    WR = Wratio(p)
    if WR<dummy[1]
        return WR,p
    else
        return dummy
    end
end

#@time loopGeneral(0.01,eval,(1.0,Ex(1.0,1.0,0.0,1.0,0.0,1.0)))
#287.295988 seconds (4.52 M allocations: 230.495 MiB, 0.02% gc time)
#(0.9541720118311154, Ex(0.04, 0.04, 0.01, 1.0, 0.0, 0.99))

###############################################################################
#####lowest welfare ratio if only mech design and no info design is used#######
###############################################################################

#budget balance constraint, BBfullInfo>=0, with full info and prob of trade between vu (vb) and cu (cb) is yuu (ybb)
@everywhere BBfullInfo(p,yuu,ybb)=yuu*p.gu*(1-p.wb)*p.vu-ybb*p.wb*(1-p.gu)*p.cb+p.wb*((1-p.gu)*ybb*p.vb+p.gu*yuu*p.vu+p.gu*p.vb*(1-yuu))-p.gu*(p.wb*p.cb*ybb+(1-p.wb)*yuu*p.cu+p.cu*p.wb*(1-ybb))

@everywhere WfullInfo(p,yuu,ybb)=p.vb*p.wb*(p.gu+(1-p.gu)*ybb)+p.vu*p.gu*(1-p.wb)*yuu-p.cu*p.gu*(p.wb+(1-p.wb)*yuu)-p.cb*(1-p.gu)*p.wb*ybb

@everywhere function welfareOptMech(p)
    if BBfullInfo(p,1.0,1.0)<-1e-6
        yuuInt = -(p.wb*(1-p.gu)*(p.vb-p.cb)+p.wb*p.gu*(p.vb-p.cb))/(p.gu*(1-p.wb)*(p.vu-p.cu)+p.wb*p.gu*(p.vu-p.vb))  
        ybbInt = -(p.gu*(1-p.wb)*(p.vu-p.cu)+p.wb*p.gu*(p.vu-p.cu))/(p.wb*(1-p.gu)*(p.vb-p.cb)+p.gu*p.wb*(p.cu-p.cb))
        return max(WfullInfo(p,yuuInt,1.0),WfullInfo(p,1.0,ybbInt))
    else
        return WfullInfo(p,1.0,1.0)
    end
end

@everywhere function WratioMech(p)
    W1stBest = Wfb(p)
    W2ndBest = welfareOptMech(p)
    return W2ndBest/W1stBest
end
@everywhere function evalMech(p,dummy)
    WR = WratioMech(p)
    if WR<dummy[1]
        return WR,p
    else
        return dummy
    end
end

@time loopGeneral(0.01,evalMech,(1.0,Ex(1.0,1.0,0.0,1.0,0.0,1.0)))
# 257.491181 seconds (11.36 k allocations: 384.422 KiB)
# (0.8918918918918913, Ex(0.04, 0.04, 0.02, 1.0, 0.01, 0.99))
