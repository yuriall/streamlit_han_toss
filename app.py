import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title of the dashboard
st.title('Sample Streamlit Dashboard')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
n = st.sidebar.slider('Number of data points', 10, 100, 50)
option = st.sidebar.selectbox('Select a chart type', ('Line Chart', 'Bar Chart'))

# Generate random data
data = pd.DataFrame({
    'x': np.arange(n),
    'y': np.random.randn(n).cumsum()
})

# Display the data
st.write('### Generated Data', data)

# Plot the data
st.write('### Chart')
if option == 'Line Chart':
    st.line_chart(data.set_index('x'))
elif option == 'Bar Chart':
    fig, ax = plt.subplots()
    ax.bar(data['x'], data['y'])
    st.pyplot(fig)

# Adding a map
st.write('### Map')
map_data = pd.DataFrame(
    np.random.randn(100, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])
st.map(map_data)