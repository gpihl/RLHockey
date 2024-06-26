import numpy as np




import matplotlib.pyplot as plt

x = np.linspace(0, 1, 100)
y = smoothstep(x)

plt.plot(x, y)
plt.plot(x, x, '--')  # Plot the identity function for comparison
plt.xlabel('x')
plt.ylabel('y')
plt.title('Smoothstep vs Identity')
plt.legend(['Smoothstep', 'Identity'])
plt.grid(True)
plt.show()