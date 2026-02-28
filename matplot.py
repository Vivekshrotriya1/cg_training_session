# # import matplotlib.pyplot as plt

# # months = ['jan', 'feb', 'mar', 'apr']
# # sales = [1000, 2000, 1500, 1300]

# # plt.plot(months, sales)
# # plt.show()

# # import matplotlib.pyplot as plt

# # months = ["January", "February", "March", "April"]
# # attendance = [85, 90, 95, 88]

# # plt.plot(months, attendance, marker="o")
# # plt.title("Student Attendance Trend (First Four Months)")
# # plt.xlabel("Month")
# # plt.ylabel("Average Students Present")
# # plt.grid(True)

# # plt.show()

# import matplotlib.pyplot as plt

# sections = ["North Wing", "South Wing", "East Wing", "West Wing"]
# books_checked_out = [2000, 1500, 1800, 2200]

# plt.bar(sections, books_checked_out)
# plt.title("Library Book Checkouts by Section (Last Semester)")
# plt.xlabel("Library Sections")
# plt.ylabel("Number of Books Checked Out")

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# # simulate salary data
# salaries = np.random.normal(50000, 10000, 1000)

# plt.hist(salaries, bins=20)
# plt.title("Distribution of Employee Salaries")
# plt.xlabel("Salary")
# plt.ylabel("Number of Employees")

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # generate exam scores
# scores = np.random.normal(75, 10, 1000)

# plt.hist(scores, bins=20)
# plt.title("Distribution of Statistics Exam Scores")
# plt.xlabel("Marks")
# plt.ylabel("Number of Students")

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# regions=["North","South","East","West"]
# revenue=[128000,94000,155000,112000]
# plt.bar(regions,revenue,color=["skyblue","orange","green","red"])

# plt.ylabel('Revenue ($)')
# plt.title('Q2 Sales by Revenue')
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # example campaign data (ad spend in dollars)
# ad_spend = np.array([2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000])

# # simulated sales revenue (example pattern: sales increase with ad spend)
# sales = np.array([15000, 20000, 25000, 30000, 36000, 42000, 47000, 52000, 58000])

# # scatter plot
# plt.scatter(ad_spend, sales)

# # create trend line
# m, b = np.polyfit(ad_spend, sales, 1)
# plt.plot(ad_spend, m * ad_spend + b)

# plt.title("Relationship Between Advertising Spend and Sales")
# plt.xlabel("Advertising Spend ($)")
# plt.ylabel("Sales Revenue ($)")

# plt.show()

# import random

# heads=0
# tot=1000

# for i in range(tot):
#                res=random.randint(0,1)
#                if res==1:
#                        heads=heads+1

# tails=tot-heads;

# print("Heads: ",heads)
# print("Tails: ",tails)
# print("p(heads)= ",heads/tot)

# import numpy as np
# import matplotlib.pyplot as plt

# heights=np.random.normal(165,8,200)

# print("Shortest: ",round(heights.min(),1))
# print("Maximum: ",round(heights.max(),1))
# print("Average: ",round(heights.mean(),1))

# #Draw the ball shape
# plt.hist(heights,bins=20,color='#00BFA6',edgecolor='white')


# plt.axvline(heights.mean(),color='red',linewidth=2,label='Average')
# plt.title('Heights of 200 Students')
# plt.xlabel('Height (cm)')
# plt.ylabel("Number of students")
# plt.show()


from scipy.stats import binom

n=10
p=0.25

p3=binom.pmf(3,n,p)

#P (exaclty 3 correct)?
print("P(exactly 3)",round(p3,3))

# #P (exaclty 5 correct)?
# p5=binom.pmf(5,n,p)
# print("P(exactly 5)",round(p5,3))



#P (atleat 5 correct)?
# p5=1-binom.cdf(4,n,p)
# print("P(5 or more)",round(p5,3))

# import matplotlib.pyplot as plt
# import numpy as np

# k=np.arange(0,11)
# plt.bar(k,binom.pmf(k,n,p),color='pink',edgecolor='white')
# plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('housing_dataset.csv')

x=df['size','bedrooms','age','location']
y=df['price']

x_train,x_test,y_test,y_val = train_test_split(
               x_train,y_train,
)