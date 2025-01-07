import streamlit as st

def main():
    # Page Title
    st.set_page_config(page_title="Home - Data Science Project", page_icon="🏠")
    st.title("🏠 Welcome to My Data Science Project")

    # Welcome Message
    st.markdown(
        """
        ### 🔍 Explore the fascinating world of data with us!
        - Analyze **user behavior** through detailed insights.
        - Build and explore **predictive models** for real-world applications.
        - Uncover **trends and relationships** between features.
        """
    )

    # Key Sections
    st.header("📑 Key Sections in This Project")
    st.markdown(
        """
        - **Introduction**: Get an overview of the dataset and project objectives.
        - **EDA**: Explore the data through visualizations and insights.
        - **Modeling**: Dive into predictive models for battery drain and more.
        - **Conclusion**: Discover actionable insights and fun facts!
        """
    )

    # Navigation Help
    st.info(
        """
        Use the **sidebar menu** to navigate between sections.  
        Start with the **Introduction** to learn about the dataset and project goals.
        """
    )

    # Useful Links
    st.header("🔗 Useful Links")
    st.markdown(
        """
        - [📂 GitHub Repository](https://github.com/affanabid)  
        - [📄 Documentation](https://docs.streamlit.io/)  
        - [📧 Contact Me](mailto:affanabid31@gmail.com)
        """
    )

if __name__ == "__main__":
    main()
