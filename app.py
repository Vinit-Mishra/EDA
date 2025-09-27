import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
st.set_page_config(
    page_title="Student Performance EDA",
    layout="wide",
    page_icon="🎓"
)

# --- Data Loading ---
@st.cache_data
def load_data():
    # Load the dataset (assuming it's in the same directory)
    data = pd.read_csv("StudentsPerformance.csv")
    # Clean column names for easier access (e.g., 'math score' -> 'math_score')
    data.columns = data.columns.str.replace(' ', '_').str.lower()
    return data

df = load_data()

st.title("👨‍🎓 Student Performance Exploratory Data Analysis (EDA)")
st.caption("An interactive dashboard built with Streamlit and your 'StudentsPerformance.csv' data.")

# --- Sidebar for Filtering ---
st.sidebar.header("Data Filters & Analysis Options")

# Display the raw data toggle
if st.sidebar.checkbox('Show Raw Data', False):
    st.subheader('Raw Data (First 10 Rows)')
    st.dataframe(df.head(10))

# --- Main Dashboard Sections ---

# 1. Data Summary
st.header("1. Data Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Students", df.shape[0])

with col2:
    st.metric("Total Features", df.shape[1])

with col3:
    if df.isnull().sum().sum() == 0:
        st.success("✅ No Missing Values Found.")
    else:
        st.warning("⚠️ Missing Values Detected.")

st.subheader("Descriptive Statistics")
st.dataframe(df.describe().T) # Transpose for better view

st.markdown("---")

## 2. Univariate Analysis (Distributions)
st.header("2. Score Distributions")
score_type = st.selectbox(
    "Select a Score to Visualize its Distribution:",
    ['math_score', 'reading_score', 'writing_score']
)

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
plt.subplots_adjust(wspace=0.3) # Adjust space between plots

# Histogram
sns.histplot(df[score_type], kde=True, ax=ax[0], bins=20, color='skyblue')
ax[0].set_title(f'Distribution of {score_type.replace("_", " ").title()}', fontsize=14)

# Box Plot
sns.boxplot(y=df[score_type], ax=ax[1], color='lightcoral')
ax[1].set_title(f'Box Plot of {score_type.replace("_", " ").title()} (Outlier Check)', fontsize=14)

st.pyplot(fig)
st.markdown("---")

## 3. Categorical Impact Analysis
st.header("3. Impact of Categorical Features on Scores")

category = st.selectbox(
    "Select a Categorical Feature to Analyze its Impact on Average Scores:",
    ['gender', 'race/ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
)

# Calculate mean scores for the selected category
score_mean_df = df.groupby(category)[['math_score', 'reading_score', 'writing_score']].mean().reset_index()
score_mean_df = score_mean_df.sort_values('math_score', ascending=False)

st.subheader(f"Average Scores by {category.replace('_', ' ').title()}")

# Display highlighted table
st.dataframe(score_mean_df.style.highlight_max(axis=0, color='yellow'))

# Visualization (Bar Chart)
fig, ax = plt.subplots(figsize=(12, 7))
score_mean_df.set_index(category).plot(kind='bar', ax=ax, rot=45, colormap='viridis')
ax.set_title(f'Average Scores Grouped by {category.replace("_", " ").title()}', fontsize=16)
ax.set_ylabel('Average Score')
plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

## 4. Correlation Analysis
st.header("4. Correlation Between Scores")
st.info("The correlation heatmap shows how strongly the numerical scores are related.")

# Select only the score columns
corr_df = df[['math_score', 'reading_score', 'writing_score']]
correlation_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    linewidths=.5, 
    linecolor='black',
    ax=ax
)
ax.set_title("Correlation Heatmap of Student Scores", fontsize=14)
st.pyplot(fig)