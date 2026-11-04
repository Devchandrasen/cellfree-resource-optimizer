# Methods and solver behaviour

All methods optimise minimum delivered service using the same channel moments, payload-forwarding model and physical constraints. They use independent cold starts. No method receives another method's solution.

## Joint allocation

`joint` builds a capacity-aware coherent starting point, maintains separate useful endpoints and alternates coherent AP refinements with coupled mode and association proposals. Fixed-graph power and payload subproblems use reusable conic kernels. Structured cases receive specialised checks. Each proposed allocation must pass the physical model before it can replace the incumbent.

## Coherent controls

`strong-cjt` spends its budget on coherent association and power refinement. `mode-informed-cjt` also projects useful mode-related proposals back to a coherent graph. Both must return only coherent user modes.

## Noncoherent and adaptive pure modes

`pure-ncjt` uses an independent noncoherent start, aggregate radio guidance, AP refinements and payload recovery under the fixed successive-cancellation order.

`adaptive-pure` allocates time to coherent search and conditionally evaluates noncoherent service. It returns one pure serving mode for the complete network. The final choice is made inside its own budget.

## Hybrid controls

`fixed-hybrid` chooses each user's strongest large-scale links once, then optimises modes and power on that fixed graph. The graph contains exactly `max_aps` selected links per user, although some links may carry zero power.

`swap-switch` evaluates opposite-mode user swaps, individual switches and AP refinements. Its power updates use successive convex approximations, positive starts and independent endpoint preselection.

## Numerical implementation

Clarabel solves conic subproblems with QDLDL and one solver thread. SciPy provides HiGHS linear and mixed-integer optimization. Reusable conic kernels avoid repeated structural assembly where possible. Typical conic limits are 120 iterations, absolute and relative gap tolerances of `1e-8` and feasibility tolerances of `1e-9` for reusable kernels or `1e-8` otherwise. Root guidance uses 90 iterations. The default outer improvement target is `4e-4` bit/s/Hz.

These are internal controller settings. The public interface exposes the network configuration, method, optimizer seed and total acceptance budget. Changing engine settings creates a different numerical method and should be recorded as a separate software revision.

A successful native status alone is not a physical feasibility certificate. Payload recovery and the independent validator check the returned allocation. Outer bounds and aggregate relaxations are guidance and do not imply global joint optimality.
