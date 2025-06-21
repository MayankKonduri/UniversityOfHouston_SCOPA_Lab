import matplotlib.pyplot as plt

from SetupNN import *

n_train = 8000
n_test = 2000
data = load_nndata("rdiffEQ-nn-spl-10000.npz", n_test, n_train )
#data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.001.npz", n_test, n_train )
#data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.01.npz", n_test, n_train )
#data = load_nndata("rdiffEQ-nn-spl-10000-addnoise-delta=0.1.npz", n_test, n_train )


t_shape = data["theta_train"].shape
y_shape = data["y_train"].shape

n = y_shape[1] - 1

print('size of training data:', n)
print('number of examples:', y_shape[0])


x = np.linspace(0, 1, n+1 )

plt.figure(figsize=(10, 6))
plt.plot(x, data["y_train"].T)
plt.xlabel(r'$x$',fontsize=20)
plt.ylabel(r'$y_{\text{obs}}$',fontsize=20)
plt.savefig("observations-10k-delta0d1.pdf", format="pdf", bbox_inches="tight")
plt.show( )
