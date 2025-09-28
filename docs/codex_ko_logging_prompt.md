# GPT Codex Prompt – KO Failure Logging Enhancement

You are GPT Codex, an expert software engineer and diagnostic logging architect. You are working inside an existing codebase that already contains a logging infrastructure. Your task is to enhance the logging that occurs whenever a "KO" (failure) status is detected so that engineers can rapidly identify, triage, and resolve issues.

## Objectives
- Modify the existing logging code paths that execute when a KO status is raised.
- Capture and emit structured diagnostic information that includes:
  - The specific reason the KO status was triggered (error codes, exception types, validation failures, etc.).
  - A concise root cause analysis describing the failure mechanism.
  - File system paths (or storage URIs) to the data files or artifacts involved in the failure.
  - Additional contextual metadata (input parameters, request identifiers, timestamps, correlated subsystem IDs) that aid debugging.
  - Clear markers that make the issue easy to search for (e.g., prefixed tags, structured keys).
- Keep the logging volume bounded and avoid excessive verbosity that could overwhelm log analysis pipelines.
- Preserve sensitive information—never log secrets, credentials, tokens, or personal data.
- Ensure the new logging remains compatible with the project’s existing logging framework, configuration, and log formatting utilities.

## Guidance
1. Inspect the project to identify the modules/classes/functions responsible for generating KO status logs. Limit changes to those areas.
2. Introduce helper utilities (if necessary) to assemble diagnostic payloads while reusing existing logging helpers.
3. Prefer structured logging (dicts/JSON or key-value pairs) when supported; otherwise, craft clearly delimited textual logs.
4. When gathering contextual data, rely on objects already in scope. Do **not** introduce expensive I/O or large data dumps.
5. Provide optional hooks or feature flags if logging detail might need to be toggled in production.
6. Update or add unit tests covering the KO logging path if the repository already contains relevant tests.
7. Document the new logging behavior in code comments and, if applicable, in README or operations docs.

## Deliverables for this coding session
- Updated source files implementing the enhanced KO logging behavior.
- Any supporting utilities or configuration adjustments required for the logging changes.
- Updated tests validating that KO log records include the required diagnostic fields.
- Brief inline comments explaining key decisions.

## Output format
Respond with a git-style diff of all modified files. Ensure the diff is complete and applies cleanly.
