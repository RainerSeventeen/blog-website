export interface RepoData {
  name: string;
  owner: string;
  stars: number;
  forks: number;
  language: string;
  license: string;
}

const fallback = (owner: string, name: string): RepoData => ({
  name,
  owner,
  stars: 0,
  forks: 0,
  language: "Unknown",
  license: "—",
});

async function fetchRepo(owner: string, name: string): Promise<RepoData> {
  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
  };
  const token =
    (typeof import.meta !== "undefined" && import.meta.env?.GITHUB_TOKEN) ||
    process.env.GITHUB_TOKEN;
  if (token) headers["Authorization"] = `Bearer ${token}`;

  try {
    const res = await fetch(
      `https://api.github.com/repos/${owner}/${name}`,
      { headers },
    );
    if (!res.ok) return fallback(owner, name);
    const json = await res.json();
    return {
      name,
      owner,
      stars: json.stargazers_count ?? 0,
      forks: json.forks_count ?? 0,
      language: json.language ?? "Unknown",
      license: json.license?.spdx_id ?? "—",
    };
  } catch {
    return fallback(owner, name);
  }
}

// ---- 整点刷新缓存 ----

const store = new Map<string, RepoData>();
let repoList: Array<{ owner: string; name: string }> = [];
let initPromise: Promise<void> | null = null;

function msUntilNextHour(): number {
  const now = new Date();
  const next = new Date(now);
  next.setHours(now.getHours() + 1, 0, 0, 0);
  return next.getTime() - now.getTime();
}

async function refreshAll(): Promise<void> {
  const label = new Date().toISOString().slice(0, 19);
  console.log(`[github-cache] refreshing ${repoList.length} repos at ${label}`);
  await Promise.allSettled(
    repoList.map(async ({ owner, name }) => {
      const data = await fetchRepo(owner, name);
      store.set(`${owner}/${name}`, data);
    }),
  );
  console.log(`[github-cache] done`);
}

function scheduleHourly(): void {
  const delay = msUntilNextHour();
  const nextTime = new Date(Date.now() + delay).toISOString().slice(11, 16);
  console.log(`[github-cache] next refresh scheduled at ${nextTime} UTC`);
  setTimeout(() => {
    refreshAll();
    setInterval(refreshAll, 60 * 60 * 1000);
  }, delay);
}

/**
 * 初始化仓库缓存。首次调用会立即拉取数据并等待完成，随后安排整点刷新。
 * 之后再次调用直接返回，不重复初始化。
 */
export async function initRepoCache(
  repos: Array<{ owner: string; name: string }>,
): Promise<void> {
  repoList = repos;
  if (!initPromise) {
    initPromise = refreshAll().then(scheduleHourly);
  }
  await initPromise;
}

/**
 * 从缓存中同步读取仓库数据。需在 initRepoCache 完成后调用。
 */
export function getRepoCached(owner: string, name: string): RepoData {
  return store.get(`${owner}/${name}`) ?? fallback(owner, name);
}
