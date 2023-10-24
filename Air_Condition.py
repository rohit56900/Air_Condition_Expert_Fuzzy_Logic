import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'D:\01 Lovely Professional University\5TH SEMESTER\PYTHON\Unit_4\input_data.csv'
data = pd.read_csv(file_path)
temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
fan_speed = ctrl.Consequent(np.arange(0, 101, 1), 'fan_speed')
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['cool'] = fuzz.trimf(temperature.universe, [0, 50, 100])
temperature['warm'] = fuzz.trimf(temperature.universe, [50, 100, 100])
humidity['dry'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['comfortable'] = fuzz.trimf(humidity.universe, [0, 50, 100])
humidity['humid'] = fuzz.trimf(humidity.universe, [50, 100, 100])
fan_speed['low'] = fuzz.trimf(fan_speed.universe, [0, 0, 50])
fan_speed['medium'] = fuzz.trimf(fan_speed.universe, [0, 50, 100])
fan_speed['high'] = fuzz.trimf(fan_speed.universe, [50, 100, 100])
rule1 = ctrl.Rule(temperature['cold'] & humidity['dry'], fan_speed['low'])
rule2 = ctrl.Rule(temperature['cool'] & humidity['comfortable'], fan_speed['medium'])
rule3 = ctrl.Rule(temperature['warm'] & humidity['humid'], fan_speed['high'])

# Create a control system
fan_speed_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fan_speed_simulation = ctrl.ControlSystemSimulation(fan_speed_ctrl)

# Select a random row from the data
random_index = np.random.randint(0, len(data))
random_data = data.iloc[random_index]

# Set input values from the selected data
fan_speed_simulation.input['temperature'] = random_data['temperature']
fan_speed_simulation.input['humidity'] = random_data['humidity']

# Calculate fan speed
fan_speed_simulation.compute()

# Get the fan speed value
resulting_fan_speed = fan_speed_simulation.output['fan_speed']

# Display the results
print(f"Temperature: {random_data['temperature']}°C")
print(f"Humidity: {random_data['humidity']}%")
print(f"Fan Speed: {resulting_fan_speed}%")

# Visualize the membership functions
temperature.view()
humidity.view()
fan_speed.view()

# Scatter plot
plt.scatter(random_data['temperature'], random_data['humidity'], color='red')
plt.title('Temperature vs. Humidity')
plt.xlabel('Temperature (°C)')
plt.ylabel('Humidity (%)')
plt.grid()

# Plot the fan speed output
fan_speed.view(sim=fan_speed_simulation)
plt.title(' Fan Speed ')
plt.xlabel('Fan Speed (%)')
plt.grid()

plt.show()
