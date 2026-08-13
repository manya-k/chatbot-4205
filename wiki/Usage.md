# Usage

Starting the service
- After installation, start the service via the project start script or the commands in Installation.md.

CLI examples
- Start in development:
  - npm run dev
  - or
  - flask run

HTTP API example
- POST /api/v1/message
  - Request:
    - { "session_id": "abc", "message": "Hello" }
  - Response:
    - { "reply": "Hi, how can I help?" }

Web UI
- If a web UI is included, open http://localhost:8000

Configuration options that affect runtime
- MODEL_PROVIDER - model backend to use
- API_KEY - provider API key
- LOG_LEVEL - debug, info, warn, error

Extending connectors
- Add new connector modules under connectors/
- Implement the required interface: connect(), send_message(), close()
