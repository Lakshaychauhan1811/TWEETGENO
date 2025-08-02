# Tweet Generator

A Flask web app that uses Groq LLMs to generate viral, funny tweets from any topic or uploaded image. Connect your X (Twitter) account to post tweets directly. Supports iterative tweet improvement using AI feedback.

## Features

- Generate original, humorous tweets from any topic
- Upload images to generate tweets based on image content
- Connect your X account via OAuth 2.0 and post tweets
- Iterative tweet improvement with AI feedback
- Secure image upload and compression

## Getting Started

### Prerequisites

- Python 3.9+
- Groq API Key
- X (Twitter) Developer credentials (`CLIENT_ID`, `CLIENT_SECRET`)

### Installation

1. Clone the repository:

     ```sh
    git clone https://github.com/yourusername/tweet-generator.git
    cd tweet-generator
    ```

2. Install dependencies:

    ```sh
    pip install -r requirement.txt
    ```

3. Create a `.env` file with your API keys:

    ```
    GROQ_API_KEY=your_groq_api_key
    CLIENT_ID=your_x_client_id
    CLIENT_SECRET=your_x_client_secret
    ```

4. Run the app:

    ```sh
    python app.py
    ```

5. Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

## Usage

- Visit `/tweet` to generate tweets.
- Connect your X account to post tweets.
- Upload images for image-based tweet generation.

## File Structure

- `app.py`: Main Flask backend
- `gen.py`: Example/test script
- `index.html`: Landing page
- `TWEET.html`: Tweet generator UI
- `uploads/`: Uploaded images
- `requirement.txt`: Python dependencies
- `.env`: Environment variables

## License

MIT

---

Made for creators. Stop struggling with tweet ideas and start growing your audience!