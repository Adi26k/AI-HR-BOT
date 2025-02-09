# AI-HR-Bot

AI-HR-Bot is a Flask-based web application designed to assist with human resources tasks using artificial intelligence. This project aims to streamline HR processes and improve employee engagement through an interactive chatbot interface.

## Project Structure

```
AI-HR-Bot
├── app
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   └── templates
│       └── index.html
├── static
│   ├── css
│   │   └── styles.css
│   ├── js
│   │   └── scripts.js
│   └── images
├── config.py
├── run.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/AI-HR-Bot.git
   ```
2. Navigate to the project directory:
   ```
   cd AI-HR-Bot
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```
5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python run.py
```
The application will start on `http://127.0.0.1:5000/`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.