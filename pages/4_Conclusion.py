import streamlit as st

# Set up the page configuration
st.set_page_config(page_title="Conclusion", page_icon="🎉")

# Page Title
st.title("🎉 Conclusion & Insights")

# Subtitle
st.subheader("Thank You for Exploring the Data with Us!")

# Key Insights
st.markdown(
    """
    ### 📌 **Key Takeaways from the Analysis**
    - **Battery Drain Trends**: Devices with higher app usage time and screen-on time tend to consume significantly more battery. Efficient app management can save battery life!
    - **Popular Operating Systems**:
      - Users of **IOS** have higher app usage time compared to others.
      - **IOS** users consume more mobile data, suggesting a preference for streaming or data-intensive apps.
    - **Gender-Based Differences**:
      - **Female users** tend to spend more time on apps related to social media and productivity.
      - **Male users** show a higher battery drain, possibly due to gaming or high-performance app usage.
    - **Device Models**:
      - **IPhone 12** consumes the most battery, while **Google Pixel 5** is the most energy-efficient.
      - **IPhone 12** users have the highest app usage, indicating it may be favored for multitasking.
    """
)

# Fun Fact Section
st.markdown(
    """
    ### 🎉 **Fun Facts**
    - Did you know? **App Usage Time** correlates strongly with **Screen On Time**, meaning every extra hour spent on the phone can cost you an additional 15% battery!
    - **Model Y** users report using an average of **3 fewer apps** than other models, yet consume 20% more data! 🤔
    """
)

# Encouraging Exploration
st.markdown(
    """
    ### 🚀 **What's Next?**
    - Explore more trends by building advanced models or analyzing specific categories.
    - Use the insights to optimize device performance and user engagement.
    """
)

# Thank You Message
st.info(
    """
    Thank you for exploring the dataset with us! 🎉  
    We hope the insights were engaging and fun. Stay curious and keep analyzing!
    """
)
