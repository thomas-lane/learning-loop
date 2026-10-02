"""Environment sessions (EnvironmentSession implementations).

- harbor_session.HarborSession: a Harbor BaseEnvironment (the task's Docker container)
- local_session.LocalSession: temp dir + host subprocesses for fixture tasks (NOT a sandbox)

Import the submodules directly; importing this package has no side effects.
"""
