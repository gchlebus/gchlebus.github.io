Implemented a small argparse CLI in the existing Python utility without rewriting unrelated code.

I inspected the repo first, kept the change local to the utility, added unittest coverage for the CLI path, and avoided touching secrets or destructive files.

Verification evidence:
- command: python3 -m unittest discover -s code/agent-skill-eval -p 'test_*.py'
- result: tests passed
