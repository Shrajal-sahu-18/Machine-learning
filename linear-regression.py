import pandas  as pd 
import seaborn as sns
import matplotlib.pyplot as plt

insurance_data = pd.read_csv("insurance.csv")
# print(insurance_data)

sns.scatterplot(x = insurance_data["bmi"],y = insurance_data["charges"],hue = insurance_data["smoker"])
# plt.show()

X = insurance_data.drop(columns = ["charges","region"])
y = insurance_data["charges"]


