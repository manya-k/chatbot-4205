# Configuration

Recommended environment variables
- PORT - port to listen on, default 8000
- MODE - development or production
- MODEL_PROVIDER - name of model provider (example: local, openai)
- MODEL_NAME - model identifier
- PROVIDER_API_KEY - API key for model provider
- DATABASE_URL - persistence database connection
- LOG_LEVEL - info, debug, warn, error

Example .env
- PORT=8000
- MODE=development
- MODEL_PROVIDER=openai
- MODEL_NAME=gpt-4
- PROVIDER_API_KEY=your_api_key_here
- DATABASE_URL=sqlite:///data.db
- LOG_LEVEL=info

Configuration file
- The project may support a config file in config/ or config.yaml. Environment variables override file values.

Secrets
- Do not commit secrets to source control. Use environment variables or a secret manager.
