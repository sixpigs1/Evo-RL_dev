"""Read SwanLab project names using an API key supplied only on stdin."""
import json
import sys
import swanlab

if sys.stdin.isatty():
    from getpass import getpass

    key = getpass('SwanLab API key: ').strip()
else:
    key = sys.stdin.readline().strip()
if not key:
    raise RuntimeError('Missing API key on stdin')
api = swanlab.Api(api_key=key)
user = api.user()
projects = [{'name': p.name, 'path': p.path, 'url': p.url} for p in api.projects(user.username, all=True)]
print(json.dumps({'username': user.username, 'projects': projects}, ensure_ascii=False))
