from dfbench import VoyagerProblem, AdamGD

# Whole workflow of opimization with adam

vp = VoyagerProblem()

optimizer = AdamGD(vp)

optimizer.optimize(max_iterations=200)
