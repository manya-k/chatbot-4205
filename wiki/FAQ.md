# FAQ

Q: What model providers are supported?
A: The project includes an adapter pattern. Implemented providers will be listed in README or connectors. If none are configured, the project may provide a mock provider for local testing.

Q: How are sessions persisted?
A: Sessions are stored in the configured database. See Configuration.md for DATABASE_URL.

Q: How can I add a new messaging connector?
A: Create a module under connectors/ that implements connect(), send_message(), and close(), then register it in the connectors registry.
