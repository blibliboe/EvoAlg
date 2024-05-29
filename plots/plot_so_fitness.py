import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from the file
file_path = 'plots/data/20240529-104508.txt'
data = pd.read_csv(file_path, header=None, names=['X', 'Y'])

# Display the data to check if it was read correctly
print(data)

# Plot the data using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(data['X'], data['Y'], marker='o', label='Data Points')
plt.title('Data Plot using Matplotlib')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# Plot the data using seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='X', y='Y', data=data, marker='o', label='Data Points')
plt.title('Data Plot using Seaborn')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
