#File which defined all the algorithms. Each algorithm takes in an ABCtype


"""
    runabc(ABCsetup::ABCtype, targetdata; progress = false, verbose = false, parallel = true)

Run ABC with ABCsetup defining the algorithm and inputs to algorithm, targetdata is the data we wish to fit the model to and will be used as an input for the simulation function defined in ABCsetup. If progress is set to `true` a progress meter will be shown. Inference will be run in parallel via multithreading if `parallel = true`. The environmental variable JULIA_NUM_THREADS needs to be set prior to launching a julia session.
"""
function runabc(ABCsetup::ABCRejection, targetdata; progress = false, verbose = false, parallel = false)

  #initalize array of particles
  particlesall = Array{ParticleRejection}(undef, ABCsetup.maxsimulations)

  if progress
    p = Progress(ABCsetup.nparticles, 1, "Running ABC rejection algorithm...", 30)
  end

  if parallel
    Printf.@printf("Preparing to run in parallel on %i processors\n", nthreads())

    particles = Array{ParticleRejection}(undef, ABCsetup.maxsimulations)
    distvec = zeros(Float64, ABCsetup.maxsimulations) #store distances in an array
    i = Atomic{Int64}(0)
    cntr = Atomic{Int64}(0)
    @threads for its = 1:ABCsetup.maxsimulations

      if i[] > ABCsetup.nparticles
        break
      end

      #get new proposal parameters
      newparams = getproposal(ABCsetup.prior, ABCsetup.nparams)
      #simulate with new parameters
      dist, out = ABCsetup.simfunc(newparams, ABCsetup.constants, targetdata)
      #keep track of all particles incase we don't reach nparticles with dist < ϵ
      particlesall[its] = ParticleRejection(newparams, dist, out)

      #if simulated data is less than target tolerance accept particle
      if dist < ABCsetup.ϵ
        particles[its] = ParticleRejection(newparams, dist, out)
        distvec[its] = dist
        atomic_add!(i, 1)
      end
      atomic_add!(cntr,1)

    end
    # Remove particles that are still #undef and corresponding distances
    idx = [isassigned(particles,ii) for ii in eachindex(particles)]
    particles = particles[idx]
    distvec = distvec[idx]
    i = length(particles)    # Number of accepted particles
    its = cntr[]    # Total number of simulations

  else
    Printf.@printf("Preparing to run in serial on %i processor\n", 1)

    particles = Array{ParticleRejection}(undef, ABCsetup.nparticles)
    distvec = zeros(Float64, ABCsetup.nparticles) #store distances in an array
    i = 1 #set particle indicator to 1
    its = 0 #keep track of number of iterations
    while (i < (ABCsetup.nparticles + 1)) & (its < ABCsetup.maxsimulations)

      its += 1
      #get new proposal parameters
      newparams = getproposal(ABCsetup.prior, ABCsetup.nparams)
      #simulate with new parameters
      dist, out = ABCsetup.simfunc(newparams, ABCsetup.constants, targetdata)
      #keep track of all particles incase we don't reach nparticles with dist < ϵ
      particlesall[its] = ParticleRejection(newparams, dist, out)

      #if simulated data is less than target tolerance accept particle
      if dist < ABCsetup.ϵ
        particles[i] = ParticleRejection(newparams, dist, out)
        distvec[i] = dist
        i +=1
        if progress
          next!(p)
        end
      end
    end
    i -= 1    # Correct to total number of particels
  end

  if i < ABCsetup.nparticles
    @warn "Only accepted $(i) particles with ϵ < $(ABCsetup.ϵ). \n\tYou may want to increase ϵ or increase maxsimulations. \n\t Resorting to taking the $(ABCsetup.nparticles) particles with smallest distance"
    d = map(p -> p.distance, particlesall)
    particles = particlesall[sortperm(d)[1:ABCsetup.nparticles]]
    distvec = map(p -> p.distance, particles)
  elseif i > ABCsetup.nparticles
    particles = particles[1:ABCsetup.nparticles]
    distvec = distvec[1:ABCsetup.nparticles]
  end

  out = ABCrejectionresults(particles, its, ABCsetup, distvec)
  return out
end


function runabc(ABCsetup::ABCRejectionModel, targetdata; progress = false, verbose = false)

  ABCsetup.nmodels > 1 || error("Only 1 model specified, use ABCRejection method to estimate parameters for a single model")

  #initalize array of particles
  particles = Array{ParticleRejectionModel}(undef, ABCsetup.nparticles)

  i = 1 #set particle indicator to 1
  its = 0 #keep track of number of iterations
  distvec = zeros(Float64, ABCsetup.nparticles) #store distances in an array

  if progress
    p = Progress(ABCsetup.nparticles, 1, "Running ABC rejection algorithm...", 30)
  end

  while (i < (ABCsetup.nparticles + 1)) & (its < ABCsetup.maxsimulations)
    its += 1
    #sample uniformly from models
    model = rand(1:ABCsetup.nmodels)
    #get new proposal parameters
    newparams = getproposal(ABCsetup.Models[model].prior, ABCsetup.Models[model].nparams)
    #simulate with new parameters
    dist, out = ABCsetup.Models[model].simfunc(newparams, ABCsetup.Models[model].constants, targetdata)

    #if simulated data is less than target tolerance accept particle
    if dist < ABCsetup.ϵ
      particles[i] = ParticleRejectionModel(newparams, model, dist, out)
      distvec[i] = dist
      i +=1
      if progress
        next!(p)
      end
    end
  end

  i > ABCsetup.nparticles || error("Only accepted $(i) particles with ϵ < $(ABCsetup.ϵ). \n\tDecrease ϵ or increase maxsimulations ")
  out = ABCrejectionmodelresults(particles, its, ABCsetup, distvec)
  return out
end

function runabc(ABCsetup::ABCSMC, targetdata; verbose = false, progress = false, parallel = false)

  #run first population with parameters sampled from prior
  if verbose
    println("##################################################")
    println("Use ABC rejection to get first population")
  end
  ABCrejresults = runabc(ABCRejection(ABCsetup.simfunc, ABCsetup.nparams,
                  ABCsetup.ϵ1, ABCsetup.prior; nparticles = ABCsetup.nparticles,
                  maxsimulations = ABCsetup.maxsimulations,
                  constants = ABCsetup.constants), targetdata,
                  progress = progress, parallel = parallel);

  oldparticles, weights = setupSMCparticles(ABCrejresults, ABCsetup)
  ABCsetup.kernel.kernel_parameters = (maximum(ABCrejresults.parameters, dims = 1) - minimum(ABCrejresults.parameters, dims = 1) ./2)[:]
  ϵ = quantile(ABCrejresults.dist, ABCsetup.α) # set new ϵ to αth quantile
  ϵvec = [ϵ] #store epsilon values
  numsims = [ABCrejresults.numsims] #keep track of number of simualtions
  numpriorsims = [ABCrejresults.numsims] #keep track of number of draws from prior
  particles = Array{ParticleSMC}(undef, ABCsetup.nparticles) #define particles array

  if verbose
    println("Running ABC SMC... \n")
  end

  if parallel
    Printf.@printf("Preparing to run in parallel on %i processors\n", nthreads())
  else
    Printf.@printf("Preparing to run in serial on %i processor\n", 1)
  end

  popnum = 1
  finalpop = false

  if sum(ABCrejresults.dist .< ABCsetup.ϵT) == ABCsetup.nparticles
      @warn "Target ϵ reached with ABCRejection algorithm, no need to use ABC SMC algorithm, returning ABCRejection output..."
      return ABCrejresults
  end

  for smciter = 1:ABCsetup.maxiterations
    if (ϵ <= ABCsetup.ϵT) || (sum(numsims) >= ABCsetup.maxsimulations)
      break
    end

    if progress
      p = Progress(ABCsetup.nparticles, 1, "ABC SMC population $(popnum), new ϵ: $(round(ϵ, digits = 2))...", 30)
    end

    simulationbudget = ABCsetup.maxsimulations - sum(numsims)
    if parallel
      # Arrays initialised with length simulationbudget to enable use of unique index ii
      particles = Array{ParticleSMC}(undef, simulationbudget)
      distvec = zeros(Float64, simulationbudget)
      i = Atomic{Int64}(0)
      its = Atomic{Int64}(0)
      priorits = Atomic{Int64}(0)

      @threads for ii = 1:simulationbudget
        if i[] > ABCsetup.nparticles
          break
        end

        priorattempt = 0
        newparticle = NaN
        while priorattempt < 1e5
          priorattempt += 1
          j = wsample(1:ABCsetup.nparticles, weights)
          particle = oldparticles[j]
          newparticle = perturbparticle(particle, ABCsetup.kernel)
          priorp = priorprob(newparticle.params, ABCsetup.prior)
          if priorp > 0.0 #return to beginning of loop if prior probability is 0
            break
          end
        end
        atomic_add!(priorits, priorattempt)

        if priorattempt >= 1e5
          error("Constantly sampling particles with prior probability 0.")
        end

        #simulate with new parameters
        dist, out = ABCsetup.simfunc(newparticle.params, ABCsetup.constants, targetdata)

        #if simulated data is less than target tolerance accept particle
        if dist < ϵ
          particles[ii] = newparticle
          distvec[ii] = dist
          particles[ii].other = out
          particles[ii].distance = dist
          atomic_add!(i, 1)
        end

        atomic_add!(its,1)
      end
      i = i[]
      its = its[]
      priorits = priorits[]

      # Remove particles that are still #undef and corresponding distances
      idx = [isassigned(particles,ii) for ii in eachindex(particles)]
      particles = particles[idx]
      distvec = distvec[idx]
      if i >= ABCsetup.nparticles
          particles = particles[1:ABCsetup.nparticles]
          distvec = distvec[1:ABCsetup.nparticles]
      else # We ran out of simulation budget.
          push!(numsims, its)
          push!(numpriorsims, priorits)
          break
      end

    else
      particles = Array{ParticleSMC}(undef, ABCsetup.nparticles)
      distvec = zeros(Float64, ABCsetup.nparticles)
      i = 0
      its = 0
      priorits = 0
      while i < ABCsetup.nparticles
        j = wsample(1:ABCsetup.nparticles, weights)
        particle = oldparticles[j]
        newparticle = perturbparticle(particle, ABCsetup.kernel)
        priorp = priorprob(newparticle.params, ABCsetup.prior)
        priorits += 1
        if priorp == 0.0 #return to beginning of loop if prior probability is 0
          continue
        end

        #simulate with new parameters
        dist, out = ABCsetup.simfunc(newparticle.params, ABCsetup.constants, targetdata)
        its += 1

        #if simulated data is less than target tolerance accept particle
        if dist < ϵ
          i += 1
          particles[i] = newparticle
          particles[i].other = out
          particles[i].distance = dist
          distvec[i] = dist
          if progress
            next!(p)
          end
        end

        if its >= ABCsetup.maxsimulations
          particles = particles[1:i]
          distvec = distvec[1:i]
          break
        end
      end
    end

    particles, weights = smcweights(particles, oldparticles, ABCsetup.prior, ABCsetup.kernel)
    ABCsetup.kernel.kernel_parameters = ABCsetup.kernel.calculate_kernel_parameters(particles)
    oldparticles = particles

    if finalpop == true
      break
    end

    ϵ = quantile(distvec, ABCsetup.α)

    if ϵ < ABCsetup.ϵT
      ϵ = ABCsetup.ϵT
      push!(ϵvec, ϵ)
      push!(numsims, its)
      push!(numpriorsims, priorits)
      popnum = popnum + 1
      finalpop = true
      continue
    end

    push!(ϵvec, ϵ)
    push!(numsims, its)
    push!(numpriorsims, priorits)

    if ((( abs(ϵvec[end - 1] - ϵ )) / ϵvec[end - 1]) < ABCsetup.convergence) == true
      if verbose
        println("\nNew ϵ is within $(round(ABCsetup.convergence * 100, digits=2))% of previous population, stop ABC SMC\n")
      end
      break
    end

    popnum = popnum + 1

    if verbose
      println("##################################################")
      show(ABCSMCresults(particles, numsims, numpriorsims, ABCsetup, ϵvec))
      println("##################################################\n")
    end
  end
  if sum(numsims) >= ABCsetup.maxsimulations
    if verbose
      println("\nPassed maxsimulations=$(ABCsetup.maxsimulations), stop ABC SMC\n")
    end
  end

  out = ABCSMCresults(particles, numsims, numpriorsims, ABCsetup, ϵvec)
  return out
end

"""
    runabc(ABCsetup::ABCtype, targetdata; progress = false, verbose = false)

When the SMC algorithms are used, a print out at the end of each population will be made if verbose = true.
"""
function runabc(ABCsetup::ABCSMCModel, targetdata; verbose = false, progress = false)

  ABCsetup.nmodels > 1 || error("Only 1 model specified, use ABCSMC method to estimate parameters for a single model")

  #run first population with parameters sampled from prior
  if verbose
    println("##################################################")
    println("Use ABC rejection to get first population")
  end
  ABCrejresults = runabc(ABCRejectionModel(
            map(x -> x.simfunc, ABCsetup.Models),
            map(x -> x.nparams, ABCsetup.Models),
            ABCsetup.ϵ1,
            map(x -> x.prior, ABCsetup.Models),
            constants = map(x -> x.constants, ABCsetup.Models),
            nparticles = ABCsetup.nparticles,
            maxsimulations = ABCsetup.maxsimulations),
            targetdata);

  oldparticles, weights = setupSMCparticles(ABCrejresults, ABCsetup)
  ϵ = quantile(ABCrejresults.dist, ABCsetup.α) # set new ϵ to αth quantile
  ϵvec = [ϵ] #store epsilon values
  numsims = [ABCrejresults.numsims] #keep track of number of simulations
  numpriorsims = [ABCrejresults.numsims] #keep track of number of draws from prior
  particles = Array{ParticleSMCModel}(undef, ABCsetup.nparticles) #define particles array
  weights, modelprob = getparticleweights(oldparticles, ABCsetup)
  ABCsetup = modelselection_kernel(ABCsetup, oldparticles)

  modelprob = ABCrejresults.modelfreq

  if verbose
    println("Run ABC SMC \n")
  end

  popnum = 1
  finalpop = false

  if verbose
    println(ABCSMCmodelresults(oldparticles, numsims, ABCsetup, ϵvec))
  end

  if sum(ABCrejresults.dist .< ABCsetup.ϵT) == ABCsetup.nparticles
      @warn "Target ϵ reached with ABCRejection algorithm, no need to use ABC SMC algorithm, returning ABCRejection output..."
      return ABCrejresults
  end

  for smciter = 1:ABCsetup.maxiterations
    if (ϵ < ABCsetup.ϵT) || (sum(numsims) > ABCsetup.maxsimulations)
      break
    end

    i = 1 #set particle indicator to 1
    particles = Array{ParticleSMCModel}(undef, ABCsetup.nparticles)
    distvec = zeros(Float64, ABCsetup.nparticles)
    its = 1
    priorits = ABCsetup.nparticles

    if progress
      p = Progress(ABCsetup.nparticles, 1, "ABC SMC population $(popnum), new ϵ: $(round(ϵ, digits = 2))...", 30)
    end
    while i < ABCsetup.nparticles + 1

      #draw model from previous model probabilities
      mstar = wsample(1:ABCsetup.nmodels, modelprob)
      #perturb model
      mdoublestar = perturbmodel(ABCsetup, mstar, modelprob)
      # sample particle with correct model
      j = wsample(1:ABCsetup.nparticles, weights[mdoublestar, :])
      particletemp = oldparticles[j]
      #perturb particle
      newparticle = perturbparticle(particletemp, ABCsetup.Models[mdoublestar].kernel)
      #calculate priorprob
      priorp = priorprob(newparticle.params, ABCsetup.Models[mdoublestar].prior)

      if priorp == 0.0 #return to beginning of loop if prior probability is 0
        priorits += 1
        continue
      end

      #simulate with new parameters
      dist, out = ABCsetup.Models[mdoublestar].simfunc(newparticle.params, ABCsetup.Models[mdoublestar].constants, targetdata)

      #if simulated data is less than target tolerance accept particle
      if dist < ϵ
        particles[i] = newparticle
        particles[i].other = out
        particles[i].distance = dist
        distvec[i] = dist
        i += 1
        if progress
          next!(p)
        end
      end

      its += 1
    end

    particles, weights = smcweightsmodel(particles, oldparticles, ABCsetup, modelprob)
    weights, modelprob = getparticleweights(particles, ABCsetup)
    ABCsetup = modelselection_kernel(ABCsetup, particles)
    oldparticles = particles

    if finalpop == true
      break
    end

    ϵ = quantile(distvec, ABCsetup.α)

    if ϵ < ABCsetup.ϵT
      ϵ = ABCsetup.ϵT
      push!(ϵvec, ϵ)
      push!(numsims, its)
      push!(numpriorsims, priorits)
      popnum = popnum + 1
      finalpop = true
      continue
    end

    push!(ϵvec, ϵ)
    push!(numsims, its)
    push!(numpriorsims, priorits)

    if verbose
      println("##################################################")
      println(ABCSMCmodelresults(particles, numsims, numpriorsims, ABCsetup, ϵvec))
      println("##################################################\n")
    end

    if ((( abs(ϵvec[end - 1] - ϵ )) / ϵvec[end - 1]) < ABCsetup.convergence) == true
      println("New ϵ is within $(round(ABCsetup.convergence * 100, digits = 2))% of previous population, stop ABC SMC")
      break
    end

    popnum = popnum + 1
  end

  out = ABCSMCmodelresults(particles, numsims, numpriorsims, ABCsetup, ϵvec)
  return out
end
