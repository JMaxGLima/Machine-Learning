import seaborn as sns
import matplotlib.pyplot as plt

# Load the example planets dataset
planets = sns.load_dataset("planets")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

# Draw a point plot to visualize the orbital periods
sns.pointplot(data=planets, x="method", y="orbital_period", ax=ax1)
ax1.set_xlabel("Data source")
ax1.set_ylabel("Orbital period (earth days)")
ax1.set_title("Orbital Periods of Planets")
ax1.tick_params(axis='x', rotation=45)

# Draw a boxplot to show distribution of orbital periods by method
sns.boxplot(data=planets, x="method", y="orbital_period", ax=ax2, palette="rainbow")
ax2.set_xlabel("Data source")
ax2.set_ylabel("Orbital period (earth days)")
ax2.set_title("Distribution of Orbital Periods by Detection Method")
ax2.tick_params(axis='x', rotation=45)

# Show the plot
plt.tight_layout()
plt.show()

