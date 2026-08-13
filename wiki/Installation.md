# Installation

Clone repository
- git clone https://github.com/manya-k/chatbot-4205.git
- cd chatbot-4205

Environment
- Create a .env file at the project root with required variables. See Configuration.md for recommended variables.

Python example
- python -m venv .venv
- source .venv/bin/activate
- pip install -r requirements.txt

Node.js example
- nvm use <version>
- npm ci

Docker
- docker build -t chatbot-4205 .
- docker run -p 8000:8000 --env-file .env chatbot-4205

Notes
- If the project uses a different stack, replace the steps above. If you want, I can tailor this page to the actual project stack.
