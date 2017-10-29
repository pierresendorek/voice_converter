from params.params import get_params

import matplotlib.pyplot as plt

params = get_params()

plt.plot(params["triangle_lin_interpol"])
plt.show()