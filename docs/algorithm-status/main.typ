// Algorithm Status Table — Data
// ============================================================================
// EDIT THIS FILE to add, move, or remove algorithms.
// The rendering logic lives in template.typ — you should not need to touch it.
// ============================================================================

#import "template.typ": algorithm-status-table

// ---------------------------------------------------------------------------
// 1. Finished algorithms (on `main`, tested, production-ready)
// ---------------------------------------------------------------------------

#let finished-gradient = (
  [Adam], [NAdam], [SA-GD], [L-BFGS],
)

#let finished-evolutionary = (
  // EvoxES variants
  [CMA-ES#super[\*]], [OpenES#super[\*]], [XNES#super[\*]],
  [SeparableNES#super[\*]], [DES#super[\*]], [SNES#super[\*]],
  [ARS#super[\*]], [ASEBO#super[\*]], [PersistentES#super[\*]],
  [NoiseReuseES#super[\*]], [GuidedES#super[\*]], [ESMC#super[\*]],
  // EvoxPSO variants
  [PSO#super[\*\*]], [CLPSO#super[\*\*]], [CSO#super[\*\*]],
  [DMSPSOEL#super[\*\*]], [FSPSO#super[\*\*]],
  [SLPSOGS#super[\*\*]], [SLPSOUS#super[\*\*]],
  // Standalone
  [Random Search],
)

#let finished-dfo = ()

#let finished-surrogate = (
  [BoTorch BO], [BoTorch TuRBO], [ReSTIR],
)

#let finished-generative = (
  [VAE Sampling],
)

// ---------------------------------------------------------------------------
// 2. On a feature branch (merge-ready, 0 conflicts verified)
// ---------------------------------------------------------------------------

#let branch-gradient = (
  // Optax family
  [AdaBelief], [AdaFactor], [AdamW], [Adan],
  [Lion], [Lookahead], [Nadam], [NAG],
  [Noisy SGD], [RAdam], [RMSProp],
  [SAM], [Schedule-Free Adam],
  [SGD], [SGDM], [Sophia], [Yogi],
  // SciPy wrappers
  [BFGS], [L-BFGS-B], [Newton-CG], [Dogleg],
  [TNC], [SLSQP], [SR1], [Trust-Constr],
  [Trust-Krylov], [Trust-NCG], [Nonlinear CG],
  [COBYLA#sub[scipy]], [COBYQA],
  // Custom JAX
  [SGLD], [ASAM], [Adam→L-BFGS],
  [Entropy-SGD], [SGHMC], [ARC],
  [OGD], [OAdam], [Perturbed GD],
  [Noisy Adam], [GD w/ Restarts],
  [Gaussian Smoothing GD],
  // Future branch
  [OptaxGD#sub[generic]],
)

#let branch-evolutionary = (
  // DE family
  [DE#sub[custom]], [JADE], [SHADE], [L-SHADE],
  [CoDE], [SaDE], [jDE],
  // NES
  [PGPE], [CR-FM-NES],
  // PSO (standalone implementations)
  [CLPSO#sub[standalone]], [FIPS], [QPSO],
  // Swarm / other
  [CEM], [ABC], [GWO], [IHS],
  // Nevergrad
  [NG NGOpt], [NG OnePlusOne], [NG TBPSA],
  // CMA-ES family
  [Sep-CMA-ES], [Evosax MA-ES], [Evosax LM-MA-ES],
  [JAX (1+1)-ES], [JAX (μ,λ)-ES],
  [PyCMA CMA-ES], [PyCMA Active CMA-ES],
  [PyCMA IPOP], [PyCMA BIPOP],
  // Evosax backend (14 variants, many overlap with EvoxES)
  [EvosaxES#sub[14 variants]],
)

#let branch-dfo = (
  // Powell DFO
  [PDFO LINCOA], [PDFO NEWUOA],
  [PDFO UOBYQA], [Py-BOBYQA],
  // SciPy non-grad
  [Basin Hopping], [Dual Annealing],
  // MADS (Mesh Adaptive Direct Search)
  [OMADS MADS], [OMADS OrthoMADS],
  // Future branch
  [Nelder-Mead], [Powell], [COBYLA#sub[dfo]],
)

#let branch-surrogate = (
  [BAxUS], [SAASBO], [GEBO], [LineBO],
  [qKG], [qNEI], [REMBO],
  [HEBO], [SMAC], [TuRBO-LBFGS],
)

#let branch-generative = ()

// ---------------------------------------------------------------------------
// 3. Planned (not yet implemented)
// ---------------------------------------------------------------------------

#let planned-gradient = ()
#let planned-evolutionary = ()
#let planned-dfo = ()
#let planned-surrogate = ()
#let planned-generative = ()

// ===========================================================================
// Render — you normally don't need to change anything below this line.
// ===========================================================================

#algorithm-status-table(
  footnotes: [
    #super[\*]via `EvoxES(variant=…)` #h(1em)
    #super[\*\*]via `EvoxPSO(variant=…)` #h(1em)
  ],

  categories: (
    "Gradient-Based",
    "Evolutionary",
    "Derivative-Free",
    "Surrogate-Based",
    "Generative",
  ),

  status-rows: (
    (
      label: [*Finished* \ (on `main`)],
      cells: (
        finished-gradient, finished-evolutionary, finished-dfo,
        finished-surrogate, finished-generative,
      ),
    ),
    (
      label: [*On Branch* \ (merge-ready)],
      cells: (
        branch-gradient, branch-evolutionary, branch-dfo,
        branch-surrogate, branch-generative,
      ),
    ),
    (
      label: [*Planned*],
      cells: (
        planned-gradient, planned-evolutionary, planned-dfo,
        planned-surrogate, planned-generative,
      ),
    ),
  ),
)
