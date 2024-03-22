# Scoring Monitoring System

## Overview
This project implements a scoring monitoring system designed to automatically check for new data, assess model drift, re-train and re-deploy the model if necessary, and perform diagnostics and reporting. It's structured to automate the end-to-end process of monitoring model performance and ensuring the model is updated as new data becomes available.

## How It Works
The system is structured around a series of Python scripts that together automate the process of monitoring and updating the deployed model:

1. **Check and Read New Data**: The system checks for new data that hasn't been processed yet.
2. **Model Drift Detection**: It evaluates if there's a significant drift between the deployed model's performance and the performance on the newest data.
3. **Model Re-training and Re-deployment**: If model drift is detected, the system automatically re-trains the model with the new data and re-deploys it.
4. **Diagnostics and Reporting**: Finally, diagnostics are run on the newly deployed model, and a report is generated.

## Components
- `fullprocess.py`: Orchestrates the entire monitoring, re-training, and reporting process.
- `ingestion.py`: Handles the ingestion of new data.
- `training.py`: Manages the re-training of the model.
- `scoring.py`: Provides model scoring functionality to detect drift.
- `deployment.py`: Automates the model deployment process.
- `diagnostics.py`: Runs diagnostics on the model.
- `reporting.py`: Generates a report on the model's performance.
- `apicalls.py`: Makes API calls for external integrations.
- `config.json`: Configuration file specifying paths and settings.

## Setup and Requirements
- Python 3.x
- Libraries: pandas, sklearn, numpy, matplotlib, seaborn, flask, requests
- A Linux environment with cron for scheduling the `fullprocess.py` script.

## Usage
1. Ensure all dependencies are installed using `pip`:
   ```sh
   pip install -r requirements.txt
   ```
2. Set up your `config.json` with the correct paths.
3. Schedule `fullprocess.py` to run at your desired interval using cron:
   ```sh
   crontab -e
   # Add: */10 * * * * /usr/bin/python3 /path/to/fullprocess.py
   ```
4. Start your Flask API server if using `apicalls.py`:
   ```sh
   flask run
   ```

## Deployment
The system is designed to run in a server environment with cron for scheduling. Ensure the server has Python 3 and the necessary libraries installed.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
Distributed under the MIT License. See `LICENSE` for more information.
