# Car Price Prediction

This project is a web application that predicts the selling price of a car based on various features such as kilometers driven, fuel type, seller type, transmission type, owner type, and car age. The application uses a machine learning model trained on car data to make predictions.

## Project Structure

- `app.py`: The main application file containing the Flask web server and machine learning model code.
- `CAR DETAILS FROM CAR DEKHO.csv`: The dataset used for training the model.
- `static/style.css`: The CSS file for styling the web pages.
- `templates/index.html`: The HTML template for the home page.
- `templates/result.html`: The HTML template for displaying the prediction results.

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/insertusernamed/datascience-final
    cd <repository-directory>
    ```

2. Install the required dependencies:
    ```sh
    pip install pandas numpy scikit-learn joblib flask requests
    ```

3. Run the application:
    ```sh
    python app.py
    ```

4. Open your web browser and go to `http://127.0.0.1:5000/` to access the application.
