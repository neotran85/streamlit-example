
import streamlit as st
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import seaborn as sns

# Load dataset
data = pd.read_csv('campaign.csv', delimiter=';')
print(data.columns)
# Set default style
sns.set_style("whitegrid")


plt.figure(figsize=(10, 6))
sns.histplot(data['Income'].dropna(), kde=True, bins=30)
plt.title('Income Distribution')
plt.xlabel('Income')
plt.ylabel('Number of Customers')

# Display the plot in Streamlit
st.pyplot()