import assert from "node:assert/strict";
import { test } from "node:test";
import { setImmediate } from "node:timers/promises";
import { initRepoCache, getRepoCached } from "../src/lib/github.ts";

test("GitHub cache retains successful values on HTTP, timeout and network failures", async (t) => {
  let refresh: (() => void) | undefined;
  t.mock.method(globalThis, "setTimeout", (callback: () => void) => { refresh = callback; return 0; });
  t.mock.method(globalThis, "setInterval", () => 0);
  const signal = new AbortController().signal;
  t.mock.method(AbortSignal, "timeout", (ms: number) => {
    assert.equal(ms, 5000);
    return signal;
  });
  let mode = "success";
  t.mock.method(globalThis, "fetch", async (_url: string, options: RequestInit) => {
    assert.equal(options.signal, signal);
    if (mode === "network") throw new Error("offline");
    if (mode === "timeout") throw new DOMException("timed out", "TimeoutError");
    if (mode === "http") return new Response("rate limited", { status: 403 });
    return Response.json({ stargazers_count: mode === "zero" ? 0 : 42, forks_count: 3, language: "TypeScript" });
  });
  const repo = { owner: "example", name: "blog" };
  await initRepoCache([repo]);
  assert.equal(getRepoCached(repo.owner, repo.name).stars, 42);
  for (mode of ["http", "network", "timeout"]) {
    refresh!();
    await setImmediate();
    assert.equal(getRepoCached(repo.owner, repo.name).stars, 42);
  }
  mode = "zero";
  refresh!();
  await setImmediate();
  assert.equal(getRepoCached(repo.owner, repo.name).stars, 0);
  assert.equal(getRepoCached("unknown", "repo").stars, 0);
});
