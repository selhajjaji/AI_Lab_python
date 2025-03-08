# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:44:51 2025

@author: m
"""
#############################"pandas#########################
import pandas as pd

#######Init from dict###########
sabra_fruits= {"Banana": "Yellow","Blueberry": "Blue","Watermelon":"Green","Strawberry": "Red"}
sabra_f=pd.Series(sabra_fruits)
print(sabra_f[1:3])
sabra_f2=sabra_f[1:3]
print(sabra_f2.iloc[-1])

from datetime import datetime
########Handling time###########
sabra_amounts=[10,23,24,30]
start_time=datetime.now().replace(hour=8,minute=0,second=0,microsecond=0)
data_range=pd.date_range(start=start_time,periods=4,freq='h')
sabra_rainfall_amounts_today=pd.Series(data=sabra_amounts,index=data_range)
import matplotlib.pyplot as plt
sabra_rainfall_amounts_today.plot(kind='bar',title="Rainfall amounts over time")
plt.xlabel("time")
plt.ylabel("Rainfall amount")
plt.show()

########Pandas Multi - indexing#######
sabra_d5 = pd.DataFrame(
  {
    ("public", "birthyear"):
        {("Paris","alice"):1985, ("Paris","bob"): 1984, ("London","charles"): 1992},
    ("public", "hobby"):
        {("Paris","alice"):"Biking", ("Paris","bob"): "Dancing"},
    ("private", "weight"):
        {("Paris","alice"):68, ("Paris","bob"): 83, ("London","charles"): 112},
    ("private", "children"):
        {("Paris", "alice"):np.nan, ("Paris","bob"): 3, ("London","charles"): 0}
  }
)
print("Private columns:")
print(sabra_d5.filter(like="private"))
transposed_df=sabra_d5.transpose()
print("Transposed DataFrame:",transposed_df)
#########querying#############

people=pd.DataFrame({
    "Name": ["Bob","Alice","Charlie","Lina"],
    "Age" : ["35","25","40","29"],
    "City": ["Toronto", "Montreal", "Gatineau", "Ottawa"]       
   })
data_alice=people.query('Name == "Alice"')
print(data_alice)

##############Operations on dataframes##########
import numpy as np
sabra_grades=pd.DataFrame({
    "April": np.random.randint(0,100,4),
    "May" : np.random.randint(0,100,4),
    "June": np.random.randint(0,100,4),
    "July": np.random.randint(0,100,4)
    },
    index=["Bob","Alice","Charlie","Lina"]
    )
print(sabra_grades["April"].mean())
sabra_grades+=sabra_grades*0.02
print(sabra_grades.loc[sabra_grades["May"]> 50,"May"])
failing_students=sabra_grades.loc[sabra_grades.mean(axis=1) <50]
print(failing_students)
###############################Numpy&############################
import numpy as np
def my_function_sabra(x,y):    
    return np.int8((4*x)*(3*y))

my_function_sabra(1,2)
arrays=[np.fromfunction(my_function_sabra, (2,6)) for _ in range(3)]

b = np.arange(48).reshape(4, 12)
b[1, 2]
b[1, :]
b[:, 1] 
values_16_18 = b[1, 4:7] 
print("Extracted values 16, 17, 18:", values_16_18)
c = np.arange(24).reshape(2, 3, 4)  
print(c)
for m in c:
    print("Item:")
    print(m)

for i in range(len(c)):  # Note that len(c) == c.shape[0]
    print("Item:")
    print(c[i])
    
for i in c.flat:
    print("Item:", i)
    
for i in c.flat:
    print(i==0)

q1 = np.full((3,4), 1.0)
q2 = np.full((4,4), 2.0)
q3 = np.full((3,4), 3.0)
q4 = np.vstack((q1, q2, q3))
q4.shape
q5_sabra=np.vstack((q1,q2))
print(q5_sabra)
q5_sabra.shape
q5 = np.hstack((q1, q3))
q5.shape
try:
    q5 = np.hstack((q1, q2, q3))
except ValueError as e:
    print(e)
q7 = np.concatenate((q1, q2, q3), axis=0) 
q7.shape
q8_sabra=np.concatenate((q1,q2),axis=0)
print(q8_sabra)
t = np.arange(24).reshape(4,2,3)
t1 = t.transpose((1,2,0))
print(f"matrix {t} `\n\n\n  transpose : {t1}")
t2 = t.transpose()
t2.shape
t3 = t.swapaxes(0,1) 
t3.shape
t_sabra=np.zeros((2, 7))
t_sabra_transposed = t_sabra.T
n1 = np.arange(10).reshape(2, 5)
n2 = np.arange(15).reshape(5,3)
n1.dot(n2)
a1=np.arange(8).reshape(2, 4)
a2=np.arange(8).reshape(4, 2)
a3_sabra=a1.dot(a2)
print("\nDot product (a3_sabra):")
print(a3_sabra)

print("\nShape of a3_sabra:")
print(a3_sabra.shape)
sabra = np.arange(16).reshape(4, 4) 
try:
    sabra_inverse = np.linalg.inv(sabra)
    print("\nInverse of the array:")
    print(sabra_inverse)
except np.linalg.LinAlgError:
    print("\nThe array is singular and cannot be inverted.")
identity_matrix = np.eye(4)
print("4x4 Identity Matrix:")
print(identity_matrix)

random_matrix = np.random.rand(3, 3)
determinant = np.linalg.det(random_matrix)
print(f"\nDeterminant of the 3x3 Matrix:{determinant}")

e_sabra = np.random.rand(4, 4) 
print("\n4x4 Random Matrix (e_sabra):")
print(e_sabra)
# Calculate Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(e_sabra)
print(f"\nEigenvalues:{eigenvalues}")
print(f"\nEigenvectors:{eigenvectors}")

coefficients = np.array([
    [2, 4, 1],
    [3, 8, 2],
    [1, 2, 3]
])

constants = np.array([12, 16, 3])
solution = np.linalg.solve(coefficients, constants)
# Print the solution
print("\nSolution to the linear equations:",solution)
is_correct = np.allclose(np.dot(coefficients, solution), constants)
print("\nAre the results correct? (Using allclose):", is_correct)

#########################Matplotlib
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-4,4,1000)
z=np.linspace(-5,5,1000)
y=x**2 + z**3 +6
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="y = x^2 + z^3 + 6")
plt.title("Polynomial_Sabra")  # Replace 'Sabra' with your first name
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.show()

# Generate x values
x = np.linspace(-2, 2, 500)

# Create the 4x4 grid layout using subplot2grid
plt.figure(figsize=(12, 10))

# First row: Plot x^2
plt.subplot2grid((4, 4), (0, 0), colspan=4)
plt.plot(x, x**2, 'g--', label="y = x^2")  # Green dashed line
plt.title("y = x^2 (First Row)")
plt.legend()
plt.grid(True)

# Second row: Plot x^3 and x^4
plt.subplot2grid((4, 4), (1, 0), colspan=1)
plt.plot(x, x**3, 'y-', label="y = x^3")  # Yellow line
plt.title("y = x^3 (Second Row)")
plt.legend()
plt.grid(True)

plt.subplot2grid((4, 4), (1, 1), colspan=3)
plt.plot(x, x**4, 'r-', label="y = x^4")  # Red line
plt.title("y = x^4 (Second Row)")
plt.legend()
plt.grid(True)

# Third row: Plot x^6 and x
plt.subplot2grid((4, 4), (2, 0), colspan=1)
plt.plot(x, x**6, 'b--', label="y = x^6")  # Blue dashed line
plt.title("y = x^6 (Third Row)")
plt.legend()
plt.grid(True)

plt.subplot2grid((4, 4), (2, 1), colspan=3)
plt.plot(x, x, color='magenta', label="y = x")  # Pink line
plt.title("y = x (Third Row)")
plt.legend()
plt.grid(True)

# Fourth row: Plot x^7 spanning all columns
plt.subplot2grid((4, 4), (3, 0), colspan=4)
plt.plot(x, x**7, 'r:', label="y = x^7")  # Red dotted line
plt.title("y = x^7 (Fourth Row)")
plt.legend()
plt.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


# Generate x and y values for the graph
x = np.linspace(-4, 4, 1000)
y = x**2

# Plot the graph
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="y = x^2", color="blue")
plt.title("Beautiful Point with New Point_Sabra")  # Replace 'Sabra' with your first name
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Add the "beautiful point"
beautiful_point_x = 2
beautiful_point_y = beautiful_point_x**2
plt.scatter(beautiful_point_x, beautiful_point_y, color="red", label="Beautiful Point")
plt.text(beautiful_point_x, beautiful_point_y + 5, f"({beautiful_point_x}, {beautiful_point_y})", color="red")

# Add the "new point"
new_point_x = -2
new_point_y = new_point_x**2
plt.scatter(new_point_x, new_point_y, color="green", label="New Point_Sabra")
plt.text(new_point_x, new_point_y + 5, f"({new_point_x}, {new_point_y})", color="green")

plt.legend()
plt.grid(True)
plt.show()

# Generate random data for x and y
x = np.random.uniform(3, 100, 300)  # 300 random numbers between 3 and 100
y = np.random.uniform(3, 100, 300)

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=np.random.rand(300), alpha=0.7, s=50, cmap="viridis")  # Random colors and alpha
plt.title("Random Scatter Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.colorbar(label="Color Intensity")  # Add a color bar for visualization
plt.grid(True)
plt.show()

