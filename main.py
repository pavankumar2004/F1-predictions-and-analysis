from models.linear_regression import CustomLinearRegression
from models.logistic_regression import CustomLogisticRegression
from models.kmeans import CustomKMeans
from models.knn import CustomKNN
from utils.standard_scaler import CustomStandardScaler
from utils.label_encoder import CustomLabelEncoder
from models.decision_tree import CustomDecisionTree
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="F1 Championship Analysis",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_data
def load_data():
    """Load and clean all necessary datasets"""
    # Load datasets
    races = pd.read_csv('data/races.csv')
    results = pd.read_csv('data/results.csv')
    drivers = pd.read_csv('data/drivers.csv')
    constructors = pd.read_csv('data/constructors.csv')
    qualifying = pd.read_csv('data/qualifying.csv')
    
    # Clean results data
    results['position'] = pd.to_numeric(results['position'].replace('\\N', np.nan), errors='coerce')
    results['points'] = pd.to_numeric(results['points'].replace('\\N', np.nan), errors='coerce')
    results['grid'] = pd.to_numeric(results['grid'].replace('\\N', np.nan), errors='coerce')
    
    # Drop rows with missing values in critical columns
    results = results.dropna(subset=['position', 'points', 'grid'])
    
    return races, results, drivers, constructors, qualifying


def race_finish_prediction(results_df, drivers_df):
    """Race Finish Position Prediction using Custom Linear Regression"""
    st.header("🏁 Race Finish Position Prediction")
    
    # Prepare data
    prediction_data = results_df.merge(
        drivers_df[['driverId', 'forename', 'surname']], 
        on='driverId'
    )
    prediction_data['driver_name'] = prediction_data['forename'] + ' ' + prediction_data['surname']
    
    # Remove any remaining invalid data
    prediction_data = prediction_data.dropna(subset=['grid', 'position'])
    
    # Feature selection
    features = ['grid', 'driverId']
    X = prediction_data[features].copy()
    y = prediction_data['position'].values
    
    # Handle categorical variables
    le = CustomLabelEncoder()
    X['driverId'] = le.fit_transform(X['driverId'])
    X = X.values  # Convert to numpy array
    
    # Split data
    train_idx = np.random.rand(len(X)) < 0.8
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]
    
    # Train model
    model = CustomLinearRegression(learning_rate=0.2, n_iterations=10000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - predictions) ** 2)
    r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    # Display metrics
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R² Score", f"{r2:.2f}")
    
    # Visualization
    fig = px.scatter(
        x=y_test, y=predictions,
        labels={'x': 'Actual Position', 'y': 'Predicted Position'},
        title='Predicted vs Actual Race Positions'
    )
    fig.add_trace(go.Scatter(x=[0, 20], y=[0, 20], mode='lines', name='Perfect Prediction'))
    st.plotly_chart(fig)
    
    # Interactive prediction
    st.subheader("Try a Prediction")
    grid_pos = st.slider("Starting Grid Position", 1, 20, 10)
    selected_driver = st.selectbox("Select Driver", prediction_data['driver_name'].unique())
    driver_id = prediction_data[prediction_data['driver_name'] == selected_driver]['driverId'].iloc[0]
    
    # Make prediction
    input_data = np.array([[grid_pos, le.transform([driver_id])[0]]])
    predicted_position = model.predict(input_data)[0]
    st.success(f"Predicted Finish Position: {predicted_position:.1f}")

def podium_prediction(results_df, drivers_df):
    """Podium Finish Prediction using Custom  Decision Trees"""
    st.header("🏆 Podium Finish Prediction")

    # Prepare data
    podium_data = results_df.merge(
        drivers_df[['driverId', 'forename', 'surname']],
        on='driverId'
    )
    podium_data = podium_data.dropna(subset=['position'])
    podium_data['position'] = pd.to_numeric(podium_data['position'], errors='coerce')
    podium_data['podium'] = (podium_data['position'] <= 3).astype(int)

    # Features
    features = ['grid', 'driverId']
    X = podium_data[features].copy()
    y = podium_data['podium'].values

    # Handle categorical variables
    le = CustomLabelEncoder()
    X['driverId'] = le.fit_transform(X['driverId'])
    X = X.values

    # Split data
    train_idx = np.random.rand(len(X)) < 0.8
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]

    # Train Custom Decision Tree
    model_tree = CustomDecisionTree(max_depth=5)
    model_tree.fit(X_train, y_train)
    predictions_tree = model_tree.predict(X_test)

    # Calculate accuracy for Decision Tree
    accuracy_tree = np.mean(predictions_tree == y_test)
    st.metric("Decision Tree Accuracy", f"{accuracy_tree:.2%}")

    # Interactive prediction
    st.subheader("Predict Podium Chances")
    grid_pos = st.slider("Starting Position", 1, 20, 5)
    selected_driver = st.selectbox("Driver", podium_data['forename'].unique())
    driver_id = podium_data[podium_data['forename'] == selected_driver]['driverId'].iloc[0]

    # Make prediction with Logistic Regression
    input_data = np.array([[grid_pos, le.transform([driver_id])[0]]])

    # Make prediction with Decision Tree
    podium_probability_tree = model_tree.predict(input_data)[0]
    st.success(f"Decision Tree Prediction of Podium Finish: {'Podium' if podium_probability_tree else 'Not Podium'}")


def driver_clustering(results_df, drivers_df):
    """Enhanced Driver Performance Clustering with Customizable Features and Clustering Options"""
    st.header("📊 Driver Performance Clusters")

    # Select features for clustering
    st.sidebar.subheader("Select Features for Clustering")
    features = st.sidebar.multiselect("Features", ['points', 'position', 'grid'], default=['points', 'position', 'grid'])

    # Prepare data
    driver_stats = results_df.groupby('driverId').agg({
        'points': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'position': lambda x: pd.to_numeric(x, errors='coerce').mean(),
        'grid': lambda x: pd.to_numeric(x, errors='coerce').mean()
    }).reset_index()
    
    # Remove drivers with insufficient data
    driver_stats.dropna(inplace=True)

    # Merge with driver names
    driver_stats = driver_stats.merge(
        drivers_df[['driverId', 'forename', 'surname']], 
        on='driverId', how='inner'
    )
    driver_stats['driver_name'] = driver_stats['forename'] + ' ' + driver_stats['surname']

    # Optional Standardization
    if st.sidebar.checkbox("Standardize Features", value=True):
        scaler = CustomStandardScaler()
        X = scaler.fit_transform(driver_stats[features].values)
    else:
        X = driver_stats[features].values

    # Clustering
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 4)
    kmeans = CustomKMeans(n_clusters=n_clusters, random_state=42)
    driver_stats['cluster'] = kmeans.fit_predict(X)

    # 3D visualization
    if len(features) >= 3:
        fig = px.scatter_3d(
            driver_stats,
            x=features[0],
            y=features[1],
            z=features[2],
            color='cluster',
            hover_data=['driver_name'],
            title='Driver Performance Clusters',
            labels={features[0]: f'Average {features[0].capitalize()}', 
                    features[1]: f'Average {features[1].capitalize()}',
                    features[2]: f'Average {features[2].capitalize()}'}
        )
    else:
        fig = px.scatter(
            driver_stats,
            x=features[0],
            y=features[1] if len(features) > 1 else None,
            color='cluster',
            hover_data=['driver_name'],
            title='Driver Performance Clusters',
            labels={features[0]: f'Average {features[0].capitalize()}',
                    features[1]: f'Average {features[1].capitalize()}' if len(features) > 1 else None}
        )
    st.plotly_chart(fig)

    # Cluster analysis
    st.subheader("Cluster Analysis")
    for cluster in range(n_clusters):
        cluster_drivers = driver_stats[driver_stats['cluster'] == cluster]
        st.write(f"### Cluster {cluster}")
        st.write(f"Average {features[0].capitalize()}: {cluster_drivers[features[0]].mean():.2f}")
        st.write(f"Top Drivers: {', '.join(cluster_drivers.nlargest(3, features[0])['driver_name'])}")
        st.write("---")

def car_prediction_page():
    """F1 car prediction page"""
    st.header("🏎️ F1 Car Classification")
    st.write("Upload an image of an F1 car, and I'll predict the manufacturer!")

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Initialize classifier with error handling
    try:
        classifier = CustomLogisticRegression()
    except Exception as e:
        st.error(f"Error initializing classifier: {e}")
        return

    # Model loading with progress indicator
    with st.spinner("Loading model..."):
        try:
            if not classifier.load_model():
                st.info("First-time setup: Training the model...")
                with st.progress(0):
                    classifier.train('data/Formula One Cars')
                st.success("Model training completed!")
        except Exception as e:
            st.error(f"Error loading/training model: {e}")
            return

    # Image upload with clear instructions
    st.markdown("""
    ### Upload Instructions
    - Supported formats: JPG, JPEG, PNG
    - Image should clearly show the F1 car
    - Preferably from side or front view
    """)
    
    uploaded_file = st.file_uploader("Choose an F1 car image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            image = image.convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction with progress indicator
            with st.spinner('Analyzing image...'):
                predicted_class, confidence, probabilities = classifier.predict(image)

            # Results display
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Predicted Team: **{predicted_class}**")
                st.metric("Confidence", f"{confidence*100:.1f}%")

            # Team information with more detail
            team_info = {
                'AlphaTauri': {
                    'description': "Known for their innovative designs and technical partnerships with Red Bull.",
                    'founded': "2020 (Previously Toro Rosso)",
                    'base': "Faenza, Italy"
                },
                'Ferrari': {
                    'description': "The most successful and oldest team in F1, known for their powerful engines.",
                    'founded': "1950",
                    'base': "Maranello, Italy"
                },
                # Add other teams similarly
            }

            # Display detailed team information
            if predicted_class in team_info:
                team_data = team_info[predicted_class]
                with col2:
                    st.write("### Team Details")
                    st.write(f"**Description:** {team_data['description']}")
                    st.write(f"**Founded:** {team_data['founded']}")
                    st.write(f"**Base:** {team_data['base']}")

            # Confidence distribution visualization
            st.subheader("Prediction Confidence Distribution")
            fig = go.Figure(data=[
                go.Bar(
                    x=classifier.class_names,
                    y=probabilities * 100,
                    text=[f'{p:.1f}%' for p in probabilities * 100],
                    textposition='auto',
                    marker_color=['#FF4B4B' if i == np.argmax(probabilities) else '#1F77B4' 
                                for i in range(len(probabilities))]
                )
            ])

            fig.update_layout(
                title="Confidence for Each Team",
                xaxis_title="Team",
                yaxis_title="Confidence (%)",
                yaxis_range=[0, 100],
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different image or check if the image is corrupted.")

def driver_performance_classification(results_df, drivers_df):
    """Driver Performance Classification using Custom KNN"""
    st.header("🏎️ Driver Performance Classification")
    
    # Prepare data
    classification_data = results_df.merge(
        drivers_df[['driverId', 'forename', 'surname']], 
        on='driverId'
    )
    
    # Ensure position is numeric and create performance label
    classification_data['position'] = pd.to_numeric(classification_data['position'], errors='coerce')
    classification_data['performance'] = pd.cut(
        classification_data['position'], 
        bins=[0, 3, 10, 20],
        labels=[0, 1, 2]  # Using numeric labels for our custom implementation
    )
    
    # Drop rows with NaN values
    classification_data.dropna(subset=['grid', 'points', 'performance'], inplace=True)
    
    # Features
    features = ['grid', 'points']
    X = classification_data[features].values
    y = classification_data['performance'].astype(int).values
    
    # Split data
    train_idx = np.random.rand(len(X)) < 0.8
    X_train, X_test = X[train_idx], X[~train_idx]
    y_train, y_test = y[train_idx], y[~train_idx]
    
    # Standardize features
    scaler = CustomStandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    k = st.slider("Select Number of Neighbors (K)", 1, 20, 5)
    knn = CustomKNN(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Make predictions
    predictions = knn.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    st.metric("Model Accuracy", f"{accuracy:.2%}")
    
    # Create confusion matrix data
    performance_labels = ['Top Performer', 'Mid-Tier', 'Underperformer']
    confusion_matrix = np.zeros((3, 3))
    for true, pred in zip(y_test, predictions):
        confusion_matrix[int(true)][int(pred)] += 1
    
    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted", y="Actual"),
        x=performance_labels,
        y=performance_labels,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    st.plotly_chart(fig)
    
    # Visualization of predictions
    fig = px.scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        color=[performance_labels[int(p)] for p in predictions],
        title='KNN Driver Performance Classification',
        labels={'x': 'Grid Position (Standardized)', 
                'y': 'Points (Standardized)'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig)

def historical_analysis(races_df, results_df, drivers_df):
    """Historical F1 Analysis (No ML models needed)"""
    st.header("📈 Championship History")
    
    # Prepare data
    yearly_stats = results_df.merge(
        races_df[['raceId', 'year']], 
        on='raceId'
    ).merge(
        drivers_df[['driverId', 'forename', 'surname']], 
        on='driverId'
    )
    
    yearly_stats['points'] = pd.to_numeric(yearly_stats['points'], errors='coerce')
    yearly_stats = yearly_stats.dropna(subset=['points'])
    yearly_stats['driver_name'] = yearly_stats['forename'] + ' ' + yearly_stats['surname']
    
    # Championship winners
    champions = yearly_stats.groupby(['year', 'driver_name'])['points'].sum().reset_index()
    champions = champions.sort_values(['year', 'points'], ascending=[True, False])
    champions = champions.groupby('year').first().reset_index()
    
    # Visualization
    fig = px.line(
        champions,
        x='year',
        y='points',
        title='Championship Points Evolution',
        labels={'points': 'Points', 'year': 'Year'}
    )
    st.plotly_chart(fig)
    
    # Top drivers by era
    st.subheader("Top Drivers by Era")
    era = st.selectbox("Select Era", 
                      ['1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s'])
    
    decade = int(era[:4])
    era_stats = yearly_stats[yearly_stats['year'].between(decade, decade+9)]
    top_drivers = era_stats.groupby('driver_name')['points'].sum().sort_values(ascending=False).head(5)
    
    fig = px.bar(
        x=top_drivers.index,
        y=top_drivers.values,
        title=f'Top 5 Drivers of the {era}',
        labels={'x': 'Driver', 'y': 'Total Points'}
    )
    st.plotly_chart(fig)

def main():
    st.title("🏎️ Formula 1 Championship Analysis (1950-2024)")
    
    try:
        # Load data
        races, results, drivers, constructors, qualifying = load_data()
        
        # Sidebar navigation
        analysis_type = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Championship History",
             "Race Position Prediction (Regression)",
             "Podium Prediction (Classification)",
             "Driver Performance Clusters (Clustering)",
             "Driver Performance Classification (KNN)",
             "F1 Car Classification"]  
        )
        # Show selected analysis
        if analysis_type == "Championship History":
            historical_analysis(races, results, drivers)
        elif analysis_type == "Race Position Prediction (Regression)":
            race_finish_prediction(results, drivers)
        elif analysis_type == "Podium Prediction (Classification)":
            podium_prediction(results, drivers)
        elif analysis_type == "Driver Performance Clusters (Clustering)":
            driver_clustering(results, drivers)
        elif analysis_type == "Driver Performance Classification (KNN)":
            driver_performance_classification(results, drivers)
        elif analysis_type == "F1 Car Classification":
            car_prediction_page()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure all required CSV files are in the correct location")

if __name__ == "__main__":
    main()