# Getting Started

Prerequisites
- Git
- Docker (optional)
- Language runtime (see language-specific sections below)

Quick start (generic)
1. Clone the repository:
   - git clone https://github.com/manya-k/chatbot-4205.git
2. Change into project directory:
   - cd chatbot-4205
3. Follow the language-specific install steps in Installation.md
4. Start the service locally and open the UI or connect a client.

Language-specific setup examples

Python
- Create and activate a virtual environment:
  - python -m venv .venv
  - source .venv/bin/activate
- Install:
  - pip install -r requirements.txt
- Run:
  - export FLASK_ENV=development
  - flask run

Node.js
- Install:
  - npm install
- Run:
  - npm start

Docker
- Build and run:
  - docker build -t chatbot-4205 .
  - docker run -p 8000:8000 --env-file .env chatbot-4205
