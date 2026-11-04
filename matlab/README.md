# MATLAB validation

Each command-line run exports `validation.mat` with the channel moments and returned allocation. Validate it independently:

```matlab
addpath('matlab');
report = validate_export('runs/demo/validation.mat');
```

MATLAB does not need CVX or an optimisation toolbox for this check. Numerical optimisation runs through Python, Clarabel and SciPy's HiGHS interface. The validator recomputes radio rates, successive cancellation, AP power, forwarding load, user service and association limits without repairing the supplied allocation.

Python uses zero-based arrays. Only `order` and `rank` are converted to one-based indexing in the MATLAB export. Powers are normalised AP fractions. Capacities and delivered service are in bit/s/Hz. AP budgets are in watts.
