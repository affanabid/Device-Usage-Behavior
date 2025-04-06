import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📊")

# Streamlit app title for the EDA page
st.title("📊 Exploratory Data Analysis (EDA)")

# File path to the dataset
file_path = '../data/user_behavior_dataset.csv' 

# Load the dataset using st.cache_data
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to calculate summary statistics
def summary_statistics(data):
    summary = {}
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numerical_columns:
        summary[col] = {
            'Mean': data[col].mean(),
            'Median': data[col].median(),
            'Mode': data[col].mode()[0] if not data[col].mode().empty else None,
            'Standard Deviation': data[col].std(),
            'Variance': data[col].var(),
            'Minimum': data[col].min(),
            'Maximum': data[col].max(),
            'Count': data[col].count()
        }
    
    return pd.DataFrame(summary)

# Updated Function: Missing Value Analysis
def missing_value_analysis(data):
    # Count missing values and calculate percentages
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data)) * 100
    
    # Create a DataFrame to display results
    analysis = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage': missing_percentage
    })
    return analysis

def outlier_detection(data):
    outlier_stats = {}
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    
    for col in numerical_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        outlier_count = len(outliers)
        
        outlier_stats[col] = {
            'Outlier Count': outlier_count,
            # 'Lower Bound': lower_bound,
            # 'Upper Bound': upper_bound
        }
    
    return pd.DataFrame(outlier_stats).T

def feature_distribution(data):
    # Separate numerical and categorical columns
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns
    
    distributions = {'Numerical': numerical_columns, 'Categorical': categorical_columns}
    return distributions

def correlation_analysis(data):
    # Select only numerical columns
    numerical_columns = data.select_dtypes(include=[np.number])
    
    # Compute the correlation matrix
    correlation_matrix = numerical_columns.corr()
    
    return correlation_matrix

def grouped_aggregations(data, group_by_column, aggregation_function):
    # Columns to exclude
    excluded_columns = ['Operating System', 'Gender', 'Age', 'User ID', 'User Behavior Class']
    
    # Filter out excluded columns from the numerical columns
    included_columns = [col for col in data.select_dtypes(include=[np.number]).columns if col not in excluded_columns]
    
    # Perform grouped aggregation only on included columns
    grouped_data = data.groupby(group_by_column)[included_columns].agg(aggregation_function)
    return grouped_data

def create_visualization(data, plot_type, x_column=None, y_column=None):
    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(data[x_column], kde=True, bins=30, ax=ax)
        ax.set_title(f"Histogram of {x_column}")
    elif plot_type == "Scatter Plot":
        sns.scatterplot(x=data[x_column], y=data[y_column], ax=ax)
        ax.set_title(f"Scatter Plot: {x_column} vs {y_column}")
    elif plot_type == "Box Plot":
        sns.boxplot(x=data[x_column], ax=ax)
        ax.set_title(f"Box Plot of {x_column}")
    else:
        st.warning("Unsupported plot type selected.")

    return fig

# Load and display the dataset
data = load_data(file_path)

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Calculate and display summary statistics
st.subheader("Summary Statistics")
stats_summary = summary_statistics(data)
st.dataframe(stats_summary)

# Add download button for the dataset
st.download_button(
    label="Download Dataset",
    data=data.to_csv(index=False),
    file_name="user_behavior_dataset.csv",
    mime="text/csv"
)

# Missing Value Analysis
st.subheader("Missing Value Analysis")

missing_values = missing_value_analysis(data)

# Always display the missing value table, even if all values are 0
st.dataframe(missing_values)

if missing_values['Missing Values'].sum() == 0:
    st.success("No missing values in the dataset!")
else:
    st.info("The table shows the count and percentage of missing values for each column.")

# Outlier Detection
st.subheader("Outlier Detection")
outlier_analysis = outlier_detection(data)
if not outlier_analysis.empty:
    st.dataframe(outlier_analysis)
    st.write("The table shows the number of outliers, lower bound, and upper bound for each numerical column.")
else:
    st.write("No numerical columns available for outlier detection.")

# Feature Distribution Analysis
st.subheader("Feature Distribution Analysis")
distributions = feature_distribution(data)


# Categorical Feature Distribution
if len(distributions['Categorical']) > 0:
    # Side-by-side charts for Operating System and Gender
    if "Operating System" in distributions['Categorical'] and "Gender" in distributions['Categorical']:
        col1, col2 = st.columns(2)  # Create two columns

        # Operating System Chart
        with col1:
            st.write("Operating System")
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            data["Operating System"].value_counts().plot(kind='bar', ax=ax1)
            ax1.set_title("Operating System")
            ax1.set_ylabel("Count")
            st.pyplot(fig1)

        # Gender Chart
        with col2:
            st.write("Gender")
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            data["Gender"].value_counts().plot(kind='bar', ax=ax2)
            ax2.set_title("Gender")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)

    # Other categorical features
    for col in distributions['Categorical']:
        if col not in ["Operating System", "Gender"]:
            if col == "Device Model":  # Use pie chart for Device Model
                fig, ax = plt.subplots(figsize=(4, 4))
                data[col].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
                ax.set_ylabel("")  # Remove y-axis label for pie chart
                ax.set_title(f"Distribution of {col} ")
                st.pyplot(fig)
            else:  # Default to bar chart for other categorical features
                fig, ax = plt.subplots()
                data[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_ylabel("Count")
                st.pyplot(fig)
else:
    st.write("No categorical features found.")



# Correlation Analysis
st.subheader("Correlation Analysis")
correlation_matrix = correlation_analysis(data)

if not correlation_matrix.empty:
    st.write("### Correlation Matrix")
    st.dataframe(correlation_matrix)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.write("No numerical columns available for correlation analysis.")

# Grouped Aggregations Section in the UI
st.subheader("Grouped Aggregations")

# Select columns for grouping and aggregation
categorical_columns = data.select_dtypes(exclude=[np.number]).columns

if len(categorical_columns) > 0:
    group_by_column = st.selectbox("Select a column to group by:", categorical_columns)
    aggregation_function = st.selectbox("Select an aggregation function:", ['sum', 'mean', 'count', 'max', 'min'])

    # Perform grouped aggregation
    if group_by_column and aggregation_function:
        grouped_data = grouped_aggregations(data, group_by_column, aggregation_function)
        st.write(f"Grouped data by `{group_by_column}` with `{aggregation_function}` function:")
        st.dataframe(grouped_data)
else:
    st.write("No categorical columns to group by or numerical columns to aggregate.")

# Visualizations
st.subheader("Visualizations")

# Select the type of plot
# Exclude User ID from selectable numerical columns
excluded_columns = ["User ID", "User Behavior Class"]
numerical_columns = [col for col in data.select_dtypes(include=[np.number]).columns if col not in excluded_columns]

plot_type = st.selectbox("Select a plot type:", ["Histogram", "Scatter Plot", "Box Plot"])

# Select columns for the plot
if plot_type == "Histogram" or plot_type == "Box Plot":
    x_column = st.selectbox("Select a column for the plot:", numerical_columns)
    if x_column:
        fig = create_visualization(data, plot_type, x_column=x_column)
        st.pyplot(fig)
elif plot_type == "Scatter Plot":
    x_column = st.selectbox("Select the X-axis column:", numerical_columns)
    y_column = st.selectbox("Select the Y-axis column:", data.select_dtypes(include=[np.number]).columns)
    if x_column and y_column:
        fig = create_visualization(data, plot_type, x_column=x_column, y_column=y_column)
        st.pyplot(fig)
else:
    st.write("Select a valid plot type to visualize the data.")


# Pairwise Analysis
st.subheader("Insights from Relationships Between Features (Pairwise Analysis)")

# Select numerical columns for pairwise analysis
numerical_columns = data.select_dtypes(include=[np.number]).columns
selected_features = st.multiselect(
    "Select numerical features for pairwise analysis:",
    options=numerical_columns,
    default=numerical_columns
)

if len(selected_features) > 1:
    # Correlation Heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = data[selected_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

else:
    st.write("Please select at least two numerical features to perform pairwise analysis.")

# Trend Analysis
st.subheader("Trend Analysis")

# Select feature for trend analysis
numerical_columns = data.select_dtypes(include=[np.number]).columns
categorical_columns = data.select_dtypes(exclude=[np.number]).columns

# Select categorical and numerical features
trend_categorical = st.selectbox("Select a categorical feature for trend grouping:", categorical_columns)
trend_numerical = st.selectbox("Select a numerical feature to analyze trends:", numerical_columns)

if trend_categorical and trend_numerical:
    st.write(f"### Trend of `{trend_numerical}` across `{trend_categorical}`")
    
    # Calculate the mean of the numerical feature for each category
    trend_data = data.groupby(trend_categorical)[trend_numerical].mean().reset_index()
    
    # Plot the trend
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=trend_data, x=trend_categorical, y=trend_numerical, marker='o', ax=ax)
    ax.set_title(f"Trend of {trend_numerical} across {trend_categorical}")
    ax.set_xlabel(trend_categorical)
    ax.set_ylabel(f"Average {trend_numerical}")
    plt.xticks(rotation=45)
    st.pyplot(fig)
else:
    st.write("Please select both categorical and numerical features to perform trend analysis.")
