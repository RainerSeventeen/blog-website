import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { collectRecentNotes } from "../scripts/generate-recent-notes.mjs";

test("build summaries use Git dates, exclude drafts and uncommitted files, and sort newest first", () => {
  const root = mkdtempSync(path.join(tmpdir(), "recent-notes-"));
  const git = (...args) => execFileSync("git", args, { cwd: root, stdio: "pipe" });
  const write = (name, text) => writeFileSync(path.join(root, "content/note", name), text);
  try {
    mkdirSync(path.join(root, "content/note"), { recursive: true });
    git("init");
    git("config", "user.name", "Test");
    git("config", "user.email", "test@example.invalid");
    write("Old Note.md", "---\ntitle: Old\n---\n");
    write("draft.md", "---\ntitle: Draft\ndraft: true\n---\n");
    write("index.md", "# Index\n");
    git("add", ".");
    const commit = (date) => execFileSync("git", ["-c", "commit.gpgsign=false", "commit", "-m", "fixture"], {
      cwd: root, stdio: "pipe", env: { ...process.env, GIT_AUTHOR_DATE: date, GIT_COMMITTER_DATE: date },
    });
    commit("2025-01-01T00:00:00Z");
    write("C++标准容器.md", "# New `article`\n");
    write("renamed.md", "---\ntitle: Custom\nslug: custom/path\n---\n");
    git("add", ".");
    commit("2025-02-01T00:00:00Z");
    write("untracked.md", "# Untracked\n");
    const notes = collectRecentNotes(root);
    assert.equal(notes.length, 3);
    assert.equal(notes.find((note) => note.title === "New article").href, "/c标准容器/");
    assert.equal(notes.find((note) => note.title === "Custom").href, "/custom/path/");
    assert.equal(notes.at(-1).href, "/old-note/");
    assert.equal(notes[0].updated, "2025-02-01T00:00:00.000Z");
    assert.equal(notes.at(-1).published, "2025-01-01T00:00:00.000Z");
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
