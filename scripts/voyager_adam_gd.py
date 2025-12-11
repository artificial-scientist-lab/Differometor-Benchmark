from dfbench import VoyagerProblem, AdamGD

# Whole workflow of opimization with adam

vp = VoyagerProblem()

optimizer = AdamGD(vp, max_iterations=200)

optimizer.optimize()
