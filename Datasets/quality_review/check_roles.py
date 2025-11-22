#!/usr/bin/env python3
import json
import sys

for i, line in enumerate(sys.stdin, 1):
    try:
        data = json.loads(line.strip())
        convos = data.get('conversations', [])

        # Check if roles alternate properly
        roles = [msg['role'] for msg in convos]

        # Check for issues
        issues = []

        # Must start with user
        if roles and roles[0] != 'user':
            issues.append(f'starts with {roles[0]} instead of user')

        # Check for consecutive same roles
        for j in range(len(roles) - 1):
            if roles[j] == roles[j+1]:
                issues.append(f'consecutive {roles[j]} at positions {j} and {j+1}')

        # Check for invalid roles
        valid_roles = {'user', 'assistant', 'system'}
        for j, role in enumerate(roles):
            if role not in valid_roles:
                issues.append(f'invalid role {role} at position {j}')

        if issues:
            print(f'Line {i}: {" | ".join(issues)}')
            print(f'  Roles: {roles}')
            print()
    except Exception as e:
        print(f'Line {i}: Error parsing - {e}')
        print()
