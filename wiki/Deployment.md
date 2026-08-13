# Deployment

Docker
- Build:
  - docker build -t chatbot-4205 .
- Run:
  - docker run -p 8000:8000 --env-file .env chatbot-4205

Kubernetes
- Provide a Deployment manifest and Service.
- Mount secrets via Secret objects.

CI/CD
- Recommended steps:
  - Run tests and linters.
  - Build a container image.
  - Push to a registry.
  - Deploy to staging, run smoke tests, then deploy to production.

Monitoring
- Capture logs centrally.
- Monitor latency and error rates on model calls.
