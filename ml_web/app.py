import streamlit as st
import os

# Import modules from each component
from data_upload import uploader
from data_validation import statistics, visualization
from data_preprocessing import missing_values, scaling, encoding
from model_training import classification, regression, evaluation

# Set page configuration
st.set_page_config(
    page_title="ML Application",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open(os.path.join(os.path.dirname(__file__), "styles.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Upload", "Data Validation", "Data Preprocessing", "Model Training"])

# Home page
if page == "Home":
    st.title("Thiên Bảo đẹp trai vip pro")
    st.write("""
    ### Chúc mừng được vào nhóm Thiên Bảo đẹp trai sieucapvippro!
    
    This application helps you to:
    - Upload or select existing datasets
    - Validate and visualize data
    - Preprocess data for machine learning
    - Train various machine learning models
    
    Use the sidebar to navigate through different sections.
    """)
    
    st.image("https://miro.medium.com/max/1400/1*cG6U1qstYDijh9bPL42e-Q.jpeg", width=600)

# Data Upload page
elif page == "Data Upload":
    st.title("Data Upload")
    uploader.render_upload_page()

# Data Validation page
elif page == "Data Validation":
    st.title("Data Validation and EDA")
    
    if st.session_state.data is None:
        st.warning("Please upload or select data first!")
    else:
        statistics.render_statistics(st.session_state.data)
        visualization.render_visualizations(st.session_state.data)

# Data Preprocessing page
elif page == "Data Preprocessing":
    st.title("Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload or select data first!")
    else:
        data = st.session_state.data.copy()
        
        # Handle missing values
        st.subheader("Missing Values Handling")
        data = missing_values.handle_missing_values(data)
        
        # Feature scaling
        st.subheader("Feature Scaling")
        data = scaling.scale_features(data)
        
        # Encode categorical variables
        st.subheader("Categorical Encoding")
        data = encoding.encode_categorical(data)
        
        # Target selection and finalization
        st.subheader("Target Variable Selection")
        
        target_column = st.selectbox("Select target variable:", data.columns.tolist())
        
        if st.button("Apply Preprocessing"):
            st.session_state.X = data.drop(columns=[target_column])
            st.session_state.y = data[target_column]
            st.session_state.preprocessed_data = data
            st.success("Data preprocessing completed!")
            
            st.write("### Preprocessed Data Preview")
            st.write(data.head())

# Model Training page
elif page == "Model Training":
    st.title("Model Training")
    
    if st.session_state.X is None or st.session_state.y is None:
        st.warning("Please complete data preprocessing first!")
    else:
        # Determine problem type (classification or regression)
        problem_type = evaluation.determine_problem_type(st.session_state.y)
        
        if problem_type == "Classification":
            classification.render_classification_page(st.session_state.X, st.session_state.y)
        else:
            regression.render_regression_page(st.session_state.X, st.session_state.y)
