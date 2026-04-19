---
name: create-pr
description: "Automate GitHub Pull Request creation for this Astro blog project. Use when changes are committed and you need to create a PR with an auto-generated conventional-commit title and a concise description."
---

# Create PR

Automatically create a high-quality GitHub Pull Request after the current branch is ready.

## When to Use

Trigger when the user asks to create a PR, for example:
- "Create a PR"
- "Help me open a pull request"
- "Create PR for current branch"
- "Push this branch and open a draft PR"

Prerequisites:
- Code changes are committed to a git branch
- Current branch is not the default branch (`main` or `master`)
- Branch has commits not yet merged into the base branch
- Branch is pushed to the remote, or can be pushed first

## Workflow

### Step 1: Validate branch state

Use Bash to inspect the current git state:

```bash
git branch --show-current
git status -sb
git log origin/main..HEAD --oneline
git diff origin/main...HEAD --stat
git remote -v
```

Validation checklist:
- Current branch is not `main` or `master`
- Branch contains unique commits
- Base branch should usually be `main`
- If the branch is not pushed, push it before creating the PR

### Step 2: Gather PR context

Collect the information needed for a good title and body:

```bash
git log origin/main..HEAD --format="%h %s%n%b"
git diff origin/main...HEAD --stat
```

When available, use GitHub MCP tools for commit context:

```bash
mcp__github__list_commits owner=<owner> repo=<repo> sha=<current_branch>
```

Extract:
- Main purpose of the change
- Affected areas or modules
- Testing that was run
- Related issue references

### Step 3: Generate PR title and description

Title format must follow conventional commits:

```text
<type>(<scope>): <summary>
```

- Type: `feat`, `fix`, `perf`, `test`, `docs`, `refactor`, `build`, `ci`, `chore`, `revert`
- Scope is optional
- Title must be in English
- Summary must start with a lowercase letter
- Use imperative present tense
- Do not end with a period
- For breaking changes, add `!` before the colon
- To exclude from changelog, add `(no-changelog)` suffix

Validation regex:

```text
^(feat|fix|perf|test|docs|refactor|build|ci|chore|revert)(\([a-zA-Z0-9 /_-]+\))?!?: [a-z].+[^.]$
```

PR body template:

```markdown
## Summary

<1-2 paragraphs describing what changed and why>

### Changes

- <change 1>
- <change 2>
- <change 3>

## Testing

- Tests performed: <description>
OR
- Not run (not requested)
OR
- Not run (content-only changes)

## Related Issues

<issue references such as "Closes #123" or "N/A">

## Notes

<optional caveats or follow-up items>
```

### Step 4: Create the PR

Use the GitHub MCP tool:

```bash
mcp__github__create_pull_request \
  owner=<owner> \
  repo=<repo> \
  title="<conventional_commit_title>" \
  head=<current_branch> \
  base=<base_branch> \
  body="<pr_description>"
```

Optional parameters:
- `draft=true`
- `maintainer_can_modify=true`

### Step 5: Confirm result

Report:
- PR number and URL
- PR title
- Head and base branches
- Short summary of changed files or scope

## Project-Specific Conventions

For this Momo / Astro blog repository, prefer these scopes when they fit:

- `content` - blog posts, pages, and content structure
- `config` - site config and collection config
- `ui` - Astro components, layouts, and styles
- `i18n` - locale routing and translation files
- `comment` - comment components and integration
- `build` - Astro, pagefind, CI, and deployment

Examples:
- `feat(content): add related post backlinks`
- `fix(i18n): correct fallback locale route`
- `docs(config): update blog publishing guide`
- `refactor(ui): simplify post card layout`

## Error Handling

Common issues:

1. Branch not pushed
- Run `git push -u origin <branch>`

2. No unique commits
- Report that the current branch has no changes relative to the base branch

3. GitHub MCP failure
- Report the tool failure clearly and stop before inventing a PR URL

4. Title format invalid
- Rewrite the title to match the conventional-commit rules before creating the PR
