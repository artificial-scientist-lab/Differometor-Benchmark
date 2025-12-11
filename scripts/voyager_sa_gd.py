from dfbench import VoyagerProblem, SAGD

# Whole workflow of opimization with adam

vp = VoyagerProblem()

optimizer = SAGD(vp)

optimizer.optimize(wall_times=[300])
