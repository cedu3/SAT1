# Multas Prediction Project

This project is designed to predict the sum of fines imposed based on historical data. It utilizes machine learning techniques, specifically a Random Forest Regressor, to make predictions for the year 2025 based on data from previous years.

## Project Structure

The project consists of the following files and directories:

- `app.py`: The main application file that contains the Dash web application and the machine learning model for predictions.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `Dockerfile`: Contains instructions to build a Docker image for the application.
- `render.yaml`: Configuration settings for deployment on Render (optional).
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `data/MultasLimpias.xlsx`: The dataset used by the application, containing historical fine data.

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd multas-prediction
   ```

2. **Create a Virtual Environment**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```
   python app.py
   ```

   The application will be accessible at `http://127.0.0.1:8050`.

## Usage

- Use the web interface to select the months, severity levels, formats, and codes for which you want to predict the fines for 2025.
- Click on "Predecir 2025" to generate predictions and view the results in graphical format.
- You can also download the predictions as a CSV file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
# App Dash - Predicción Multas 2025

## Ejecutar local
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python app.py
