# Agent Assets

Repository-local agent guidance and reusable skills live here.

## Guidelines

- Keep repository-wide rules in the root `AGENTS.md`.
- Add subtree `AGENTS.md` files when a directory needs materially different instructions.
- Put reusable repo-local skills under `.agents/skills/<skill-name>/`.
- Prefer deterministic scripts and templates inside a skill when the same work repeats.
- Do not commit personal credentials, user-specific browser profiles, or machine-local secrets here.
- Update this README when the repository's agent workflow changes materially.
