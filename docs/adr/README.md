# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs) for the Retrieval-Free Context Compressor project.

## What are ADRs?

Architecture Decision Records are documents that capture important architectural decisions made during the development of this project, along with their context and consequences.

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-XXXX: Title

**Status:** [Proposed | Accepted | Deprecated | Superseded]
**Date:** YYYY-MM-DD
**Deciders:** [List of people involved]

## Context

What is the issue that we're seeing that is motivating this decision or change?

## Decision

What is the change that we're proposing or have agreed to implement?

## Consequences

What becomes easier or more difficult to do and any risks introduced by this change?

## Alternatives Considered

What other options were considered and why were they rejected?

## Implementation Notes

Any specific implementation details or requirements.
```

## ADR Index

| ADR | Status | Title | Date |
|-----|--------|-------|------|
| [ADR-0001](0001-compression-architecture.md) | Accepted | Hierarchical Compression Architecture | 2025-01-XX |
| [ADR-0002](0002-observability-framework.md) | Accepted | Observability and Monitoring Framework | 2025-01-XX |
| [ADR-0003](0003-testing-strategy.md) | Accepted | Comprehensive Testing Strategy | 2025-01-XX |

## Creating New ADRs

1. Copy the template from `template.md`
2. Number it sequentially (next available number)
3. Use a descriptive filename: `XXXX-short-title.md`
4. Update this index when adding new ADRs
5. Submit as part of your pull request

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)