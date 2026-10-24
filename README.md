# FairFraud

An **Explainable, Fairness-Aware Fraud Detection System** with Novel Algorithmic Enhancements on the NeurIPS BAF Benchmark.

## Project Overview
FairFraud is designed to detect financial fraud while ensuring fairness and transparency across diverse demographic groups. Utilizing the NeurIPS Bank Account Fraud (BAF) benchmark suite, this repository implements advanced machine learning techniques to optimize fairness-accuracy trade-offs alongside integrated explainable AI (XAI) tools.

## Repository Structure
```text
├── backend/          # Web application server and API logic
├── data/             # NeurIPS benchmark dataset
├── frontend/         # Web dashboard assets and UI elements
├── models/           # Pre-trained models, checkpoints, and serialization files
├── notebooks/        # Jupyter notebooks for EDA and prototyping
├── src/              # Core source code (preprocessing, pipelines, evaluation metrics)
├── .gitignore        # Files to exclude from Git tracking
├── README.md         # Repository documentation
└── requirements.txt  # Project Python dependencies
```

## Getting Started

### Prerequisites
Make sure you have Python 3.11+ installed on system.

### 1. Clone the Repository
```bash
git clone https://github.com/shubhanshutiwari1/FairFraud.git
cd FairFraud
```

### 2. Setup Environment & Install Dependencies
All necessary Python packages required to run both the models and the web application are contained in the root dependencies file:
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

### 3. Model Management
All trained predictive models are stored inside the `models/` directory at the project root. The backend application automatically pulls active checkpoints from this folder during live inference.

## Running the Web Application

To spin up the system, launch the web server directly from the `backend/` application scope.

```bash
# Navigate to the backend directory from the project root
cd backend

# Launch the web application
python app.py
```

Once started, the terminal will provide a local address (typically `http://127.0.0.1:5000` or `http://localhost:8000`). **Open this link in your web browser** to access the dashboard interface, where you can submit transactions, trigger fraud assessments, and visualize explainability metrics in real-time.
