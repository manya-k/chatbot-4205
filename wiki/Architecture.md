# Architecture

High-level components
- Web or API layer
  - Receives messages from clients and returns responses.
- Core dialog engine
  - Handles session state, routing, and conversation logic.
- Model service adapter
  - Abstracts model provider APIs and token limits.
- Connectors
  - Integrations for messaging platforms, UIs, or third-party services.
- Storage
  - Persists sessions, logs, and configurations.

Flow
1. Client sends message to API.
2. API validates and records the message.
3. Dialog engine builds a prompt and sends it to the model service adapter.
4. Model returns a response.
5. Dialog engine post-processes the response and delivers it to the client.

Scaling notes
- Keep model calls stateless where possible.
- Use caching for repeated prompts.
- Offload heavy processing to worker queues.
