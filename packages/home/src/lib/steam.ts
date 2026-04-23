export interface SteamProfileData {
  steamId64: string;
  displayName: string;
  avatar: string | null;
}

function fallback(profileUrl: string): SteamProfileData {
  const steamId64 = profileUrl.match(/profiles\/(\d+)/)?.[1] ?? "";
  return {
    steamId64,
    displayName: "Steam Community",
    avatar: null,
  };
}

function decodeXmlValue(value: string): string {
  return value
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'");
}

function extractXmlTag(xml: string, tag: string): string | null {
  const cdataMatch = xml.match(new RegExp(`<${tag}><!\\[CDATA\\[(.*?)\\]\\]><\\/${tag}>`, "s"));
  if (cdataMatch?.[1]) {
    return cdataMatch[1].trim();
  }

  const plainMatch = xml.match(new RegExp(`<${tag}>(.*?)<\\/${tag}>`, "s"));
  if (plainMatch?.[1]) {
    return decodeXmlValue(plainMatch[1].trim());
  }

  return null;
}

export async function getSteamProfile(profileUrl: string): Promise<SteamProfileData> {
  const base = fallback(profileUrl);

  try {
    const res = await fetch(`${profileUrl.replace(/\/$/, "")}/?xml=1`);
    if (!res.ok) return base;

    const xml = await res.text();
    return {
      steamId64: extractXmlTag(xml, "steamID64") ?? base.steamId64,
      displayName: extractXmlTag(xml, "steamID") ?? base.displayName,
      avatar:
        extractXmlTag(xml, "avatarFull") ??
        extractXmlTag(xml, "avatarMedium") ??
        extractXmlTag(xml, "avatarIcon"),
    };
  } catch {
    return base;
  }
}
