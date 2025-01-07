import streamlit as st

# Page title
st.title("📱 User Behavior Analysis Project")

# Dataset Description
st.header("🔍 Dataset Overview")
st.markdown(
    """
    This project utilizes a rich dataset capturing detailed user behavior metrics from mobile devices. The dataset includes:
    - **App Usage Time (min/day)**: Time spent using applications daily.
    - **Screen On Time (hours/day)**: Total screen-on duration per day.
    - **Battery Drain (mAh/day)**: Battery consumption in milliampere-hours.
    - **Number of Apps Installed**: Total apps installed on the device.
    - **Data Usage (MB/day)**: Daily internet usage in megabytes.
    - Other features such as:
      - **Device Model**: The type of device used.
      - **Operating System**: The OS powering the device.
      - **User Demographics**: Attributes like gender and age group.
    """
)

# Project Objectives
st.header("🎯 Project Objectives")
st.markdown(
    """
    This project is designed with the following objectives in mind:
    1. **Behavior Analysis**:
       - Understand how users interact with their devices by analyzing trends in app usage, screen time, and battery drain.
    2. **Data-Driven Predictions**:
       - Build predictive models to estimate **Battery Drain** and other key metrics based on user behavior.
    3. **Insights for Optimization**:
       - Provide actionable insights for device manufacturers and app developers to optimize performance and user experience.
    4. **Interactive User Experience**:
       - Create a dynamic platform for users to explore data and receive personalized predictions.
    """
)

# Key Highlights
st.header("🌟 Key Highlights")
st.markdown(
    """
    - **Comprehensive Data Analysis**: Includes detailed exploration of missing values, outliers, and correlations.
    - **Predictive Modeling**:
        - Leverages advanced techniques like linear regression to provide accurate predictions.
    - **Interactive Visualizations**:
        - Offers users the ability to explore trends, relationships, and distributions visually.
    - **Insights for Real-World Applications**:
        - Empowers decision-makers with insights to improve battery efficiency and app design.
    """
)

# Call to Action
st.info(
    """
    Ready to dive in? Use the sidebar to navigate through different sections:
    - **EDA**: Explore the dataset and uncover trends.
    - **Modeling**: Learn how predictions are made.
    - **Conclusion**: Discover actionable insights and takeaways.
    """
)
