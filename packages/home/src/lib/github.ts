export interface RepoData {
  name: string;
  owner: string;
  stars: number;
  forks: number;
  language: string;
  license: string;
}

interface CacheEntry {
  data: RepoData;
  expiresAt: number;
}

const TTL = 60 * 60 * 1000; // 1 小时
const cache = new Map<string, CacheEntry>();

async function fetchRepoFromGitHub(owner: string, name: string): Promise<RepoData> {
  const res = await fetch(`https://api.github.com/repos/${owner}/${name}`, {
    headers: { Accept: "application/vnd.github+json" },
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  const data = await res.json();
  return {
    name,
    owner,
    stars: data.stargazers_count ?? 0,
    forks: data.forks_count ?? 0,
    language: data.language ?? "Unknown",
    license: data.license?.spdx_id ?? "No License",
  };
}

export async function getRepoData(owner: string, name: string): Promise<RepoData> {
  const key = `${owner}/${name}`;
  const cached = cache.get(key);
  if (cached && Date.now() < cached.expiresAt) {
    return cached.data;
  }
  const data = await fetchRepoFromGitHub(owner, name);
  cache.set(key, { data, expiresAt: Date.now() + TTL });
  return data;
}
