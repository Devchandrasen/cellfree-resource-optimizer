# Configuration

Pass a JSON object with any of the fields below to `cellfree run --config`. Omitted fields retain their defaults. Unknown fields and inconsistent settings are rejected.

## Deployment and resources

| Field | Default | Meaning |
| :--- | ---: | :--- |
| `aps` | 14 | Number of access points |
| `antennas` | 8 | Antennas per AP |
| `users` | 10 | Number of single-antenna users |
| `max_aps` | 8 | Maximum selected APs per user, at most `aps` |
| `fronthaul` | 4.0 | Mean AP payload-forwarding capacity in bit/s/Hz |
| `fronthaul_ratio` | 7.0 | Largest to smallest AP capacity ratio when there is more than one AP |
| `ap_power` | 0.2 | Transmit power budget per AP in watts |
| `side_length` | 500.0 | Side length of the periodic square in metres |
| `height` | 10.0 | Vertical AP-to-user separation in metres |
| `seed` | 7 | Nonnegative network-generation seed |

AP capacities are geometrically spaced, normalised to the requested mean and assigned using a separate deterministic random stream. The association controllers use equal unit service weights.

## Channel and estimation settings

| Field | Default | Meaning |
| :--- | ---: | :--- |
| `pilot_power` | 0.1 | Pilot power in watts |
| `pilot_length` | 14 | Orthogonal pilot length, at least `users` |
| `coherence_symbols` | 200 | Coherence interval, strictly greater than pilot length |
| `angular_spread_degrees` | 15.0 | Gaussian local-scattering angular standard deviation |
| `quadrature_nodes` | 20 | Gauss-Hermite nodes for the spatial covariance |
| `shadow_std_db` | 4.0 | Shadow fading standard deviation in dB |
| `shadow_distance` | 50.0 | Spatial shadowing decorrelation distance in metres |
| `pathloss_intercept_db` | -30.5 | Channel-gain intercept in dB at one metre |
| `pathloss_slope` | 36.7 | Coefficient multiplying the base-10 logarithm of distance |
| `bandwidth_hz` | 20000000.0 | Receiver noise bandwidth in Hz |
| `temperature_kelvin` | 290.0 | Receiver noise temperature in kelvin |
| `noise_figure_db` | 9.0 | Receiver noise figure in dB |

The prelog is `1 - pilot_length / coherence_symbols`. Finite MMSE moments are evaluated from the channel covariance. Channel gains are divided by receiver noise power, while transmit powers remain in watts. Distances use periodic wrap-around. Plotted links are straight segments in the displayed square.

## Runtime controls

| CLI option | Default | Meaning |
| :--- | :--- | :--- |
| `--method` | `joint` | One of the seven supported methods |
| `--seconds` | 60 | Candidate acceptance budget, at least one second |
| `--seed` | 0 | Nonnegative optimizer seed for one run |
| `--plot` | off | Export a PNG and SVG overview, requires plotting extras |
| `--output` | required | New output directory |

The JSON `seed` and CLI `--seed` control different random streams. Use the JSON field for new geometries and the CLI option for repeated optimizer starts on one geometry.

For higher user loads, increase `pilot_length` as needed. For example, 20 users require at least 20 pilot symbols. The default 14-symbol setting is deliberately rejected for that load.
