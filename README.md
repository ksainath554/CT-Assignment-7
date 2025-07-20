ML Model Deployment with Streamlit
  Project Overview
    This project demonstartes the deployment of a machine learning model as an interactive web     application using Sreamlit. The application allows users to input data, receive real-time prediction from a trained model, and visualize the model's decision-making process. This serves as a practical example of making machine learning models accessible and interactive for non-technical users.

  Features-:
    1) Interactive Input: Users can adjust input features using sliders.
    2) Real-time Predictions: Get instant predictions based on the input data.
    3) Model Visulization: A scatter plot displays the synthetic training data, the model's decision boundary, and the user's input point, helping to interpret the prediction.
    4) responsive UI: Built with Streamlit, ensuring a clean and responsive user interface.
    5) Dummy Model Training: A simple Logistic Regression model is trained on synthetic data during application startup

  Technologies used-:
    1) Python 3.x
    2) Sreamlit: For building the interactive web application.
    3) Pandas: For numerical manipulation.
    4) Numpy: For numerical operations.
    5) Scikit-learn: For machine learning model training and data generation.
    6) Seaborn: For enhanced data visulization.

  How to Run Locally-: 
  Follow these steps to set up and run the application on your local machine:
    1) Clone the Repository: If you're using Git, clone your   project repository
    2) Create a Virtual Environment: It's good practice to use a virtual environment to manage project dependencies.
    python -m venv venv
    3) Activate the Virtual Environment: 
        On Windows-: .\venv\Scripts\activate
        On macOS/Linux: source venv/bin/activate
    4) Save the Application Code: Create a file named app.py in your project directory and paste the provided Streamlit code into it.
    5) Save the Requirements File: Create a file named requirements.txt in your project directory and paste the following content into it:
        streamlit
        pandas
        numpy
        scikit-learn
        matplotlib
        seaborn
    6) Install Dependencies: With your virtual environment activated, install all the required libraries:
    pip install -r requirements.txt
    7) Run the Streamlit Application: Finally, run the application from your terminal:
    streamlit run app.py
