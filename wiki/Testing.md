# Testing

Unit tests
- Tests live in tests/ or spec/
- Run tests:
  - pytest
  - or
  - npm test

Integration tests
- Use a test config that points to test fixtures.
- Consider using docker-compose for dependent services.

Test coverage
- Aim for reasonable coverage on core components: dialog engine, connectors, model adapter.

Automated checks
- Configure CI to run tests and linters on each pull request.
