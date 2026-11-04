"""Run from an installed checkout: python examples/python_api.py."""

from cellfree_optimizer import NetworkConfig, generate_network, optimize, validate_solution

config = NetworkConfig(aps=6, antennas=4, users=4, max_aps=3, pilot_length=4, seed=19)
network = generate_network(config)
result = optimize(network, method="joint", seconds=5, seed=0)
assert validate_solution(network, result.solution)["feasible"]
print(f"Minimum delivered service: {result.service:.6f} bit/s/Hz")
print(f"Coherent users: {int(result.solution['z'].sum())}/{config.users}")
