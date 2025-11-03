# Unsupervised-Mobile-RIS-MIMO

These codes are used to conduct the simulation in the submitted paper for IEEE ICC 2026.

DeepMIMO Channel Model Parameter Settings:

| DeepMIMO Dataset Parameter           | Value     |
|--------------------------------------|-----------|
| Scenario                             | O1        |
| Operating frequency                  | 60 GHz    |
| Active BSs ID                        | 3, 8      |
| Active UEs row ID                    | R5203     |
| Number of UE antennas                | 1         |
| Antenna spacing or wavelength ratio  | 0.5       |
| System bandwidth                     | 0.05 GHz  |
| Maximum number of paths              | 25        |

Simulation Parameter Settings:

| Parameter                              | Value    |
|----------------------------------------|----------|
| Number of BS antennas, M               | 4        |
| Number of users, U                     | 3        |
| Number of quantization bits            | 3 bits   |
| Transmit power at S                    | 20 dBm   |
| AWGN variance                          | -80 dBm  |
| User 1 Wobble noise variance           | -85 dBm  |
| User 2 Wobble noise variance           | -90 dBm  |
| User 3 Wobble noise variance           | -95 dBm  |
| Number of PSO particles                | 10MNU    |
| Inertia weight                         | 0.9      |
| Particle acceleration                  | 1.2      |
| Global acceleration                    | 1.2      |
| CNN model learning rate                | 0.001    |
| Kernel size at every convolution layer | 3 x 3    |
| Number of convolution layers           | 2 layers |
| Number of strides                      | 1        |
| Number of channels or kernels          | 64       |
