<script>
  import { onMount } from "svelte";

  let {
    headings = [],
    topicEntries = [],
    currentEntryId = "",
    parentFolder = null,
  } = $props();

  let activeSlug = $state("");
  let tocListElement = $state(null);
  let headingElements = $state([]);
  let ticking = false;

  const filteredHeadings = $derived(headings.filter((heading) => heading.depth === 2));
  const hasToc = $derived(filteredHeadings.length > 0);
  const hasTopicSection = $derived(Boolean(parentFolder) || topicEntries.length > 1);

  function getLinkElement(slug) {
    if (!tocListElement) {
      return null;
    }

    return Array.from(tocListElement.querySelectorAll("[data-slug]")).find((element) => element.dataset.slug === slug) || null;
  }

  function syncActiveLink() {
    if (!activeSlug || !tocListElement) {
      return;
    }

    const activeLink = getLinkElement(activeSlug);
    if (!activeLink) {
      return;
    }

    const containerTop = tocListElement.scrollTop;
    const containerBottom = containerTop + tocListElement.clientHeight;
    const linkTop = activeLink.offsetTop - 12;
    const linkBottom = linkTop + activeLink.offsetHeight + 24;

    if (linkTop < containerTop || linkBottom > containerBottom) {
      tocListElement.scrollTo({
        top: Math.max(0, activeLink.offsetTop - tocListElement.clientHeight / 2 + activeLink.offsetHeight / 2),
        behavior: "smooth",
      });
    }
  }

  function updateActiveHeading() {
    if (!headingElements.length) {
      activeSlug = "";
      return;
    }

    const anchorTop = window.scrollY + 140;
    let nextActiveSlug = headingElements[0].id;

    for (const element of headingElements) {
      if (element.offsetTop <= anchorTop) {
        nextActiveSlug = element.id;
      } else {
        break;
      }
    }

    if (activeSlug !== nextActiveSlug) {
      activeSlug = nextActiveSlug;
      syncActiveLink();
    }
  }

  function requestHeadingSync() {
    if (ticking) {
      return;
    }

    ticking = true;
    requestAnimationFrame(() => {
      updateActiveHeading();
      ticking = false;
    });
  }

  function scrollToHeading(slug) {
    const target = document.getElementById(slug);
    if (!target) {
      return;
    }

    const top = target.getBoundingClientRect().top + window.scrollY - 88;
    window.scrollTo({ top, behavior: "smooth" });
  }

  onMount(() => {
    headingElements = filteredHeadings
      .map((heading) => document.getElementById(heading.slug))
      .filter(Boolean);

    if (headingElements.length) {
      requestHeadingSync();
      window.addEventListener("scroll", requestHeadingSync, { passive: true });
      window.addEventListener("resize", requestHeadingSync);
    }

    return () => {
      window.removeEventListener("scroll", requestHeadingSync);
      window.removeEventListener("resize", requestHeadingSync);
    };
  });
</script>

{#if hasToc || hasTopicSection}
  <aside class="toc-sidebar" aria-label="文章侧栏导航">
    <div class="toc-stack">
      {#if hasTopicSection}
        <section class="toc-section topic-section" aria-labelledby="topic-heading">
          <div class="toc-heading">
            <span class="toc-heading-bar" aria-hidden="true"></span>
            <h2 id="topic-heading" class="toc-title">当前目录</h2>
          </div>

          <div class="topic-panel">
            {#if parentFolder}
              <a href={parentFolder.href} class="parent-link">
                <span class="parent-kicker">返回上级目录</span>
                <span class="parent-label">{parentFolder.label}</span>
              </a>
            {/if}

            {#if topicEntries.length > 1}
              <ul class="toc-list topic-list" role="list">
                {#each topicEntries as topicEntry, index}
                  <li>
                    <a
                      href={`/blog/${topicEntry.id}/`}
                      class:current={currentEntryId === topicEntry.id}
                      class="toc-link topic-link"
                      aria-current={currentEntryId === topicEntry.id ? "page" : undefined}
                    >
                      <span class="topic-index" aria-hidden="true">{String(index + 1).padStart(2, "0")}</span>
                      <span class="toc-text">{topicEntry.title}</span>
                    </a>
                  </li>
                {/each}
              </ul>
            {/if}
          </div>
        </section>
      {/if}

      {#if hasToc}
        <section class="toc-section" aria-labelledby="toc-heading">
          <div class="toc-heading">
            <span class="toc-heading-bar" aria-hidden="true"></span>
            <h2 id="toc-heading" class="toc-title">章节目录</h2>
          </div>

          <ul bind:this={tocListElement} class="toc-list" role="list">
            {#each filteredHeadings as heading}
              <li>
                <a
                  href={`#${heading.slug}`}
                  data-slug={heading.slug}
                  class:active={activeSlug === heading.slug}
                  class="toc-link"
                  aria-current={activeSlug === heading.slug ? "true" : undefined}
                  onclick={(event) => {
                    event.preventDefault();
                    scrollToHeading(heading.slug);
                  }}
                >
                  <span class="toc-text">{heading.text}</span>
                </a>
              </li>
            {/each}
          </ul>
        </section>
      {/if}
    </div>
  </aside>
{/if}

<style>
  .toc-sidebar {
    display: none;
  }

  @media (min-width: 1280px) {
    .toc-sidebar {
      position: fixed;
      top: 6rem;
      left: calc(50vw + var(--page-width) / 2 + 2rem - var(--article-layout-shift));
      width: min(var(--toc-panel-width), calc(100vw - (50vw + var(--page-width) / 2 - var(--article-layout-shift)) - 3rem));
      display: block;
    }
  }

  .toc-stack {
    display: flex;
    max-height: calc(100vh - 8rem);
    flex-direction: column;
    gap: 1.5rem;
    padding: 0.25rem 0 0.5rem 1.1rem;
    overflow: hidden auto;
    scrollbar-width: thin;
    scrollbar-color: var(--scrollbar-thumb) transparent;
  }

  .toc-stack::-webkit-scrollbar {
    width: 6px;
  }

  .toc-stack::-webkit-scrollbar-thumb {
    background: var(--scrollbar-thumb);
    border-radius: 999px;
  }

  .toc-section {
    position: relative;
    min-height: 0;
  }

  .toc-section::before {
    content: "";
    position: absolute;
    left: -1.1rem;
    top: 0.4rem;
    bottom: 0.2rem;
    width: 1px;
    background: color-mix(in srgb, var(--button-border-color) 78%, transparent);
  }

  .toc-heading {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    margin-bottom: 0.65rem;
  }

  .toc-heading-bar {
    display: inline-block;
    width: 0.2rem;
    height: 1rem;
    border-radius: 999px;
    background: var(--link-color);
  }

  .toc-title {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-color);
  }

  .toc-list {
    max-height: min(38vh, 24rem);
    margin: 0;
    padding: 0;
    list-style: none;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--scrollbar-thumb) transparent;
  }

  .topic-list {
    max-height: min(26vh, 16rem);
  }

  .parent-link + .topic-list {
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid color-mix(in srgb, var(--button-border-color) 72%, transparent);
  }

  .parent-link + .topic-list .toc-link {
    padding-left: 0;
  }

  .toc-list::-webkit-scrollbar {
    width: 6px;
  }

  .toc-list::-webkit-scrollbar-thumb {
    background: var(--scrollbar-thumb);
    border-radius: 999px;
  }

  .toc-link {
    display: block;
    padding: 0.35rem 0 0.35rem 0.8rem;
    border-left: 1px solid transparent;
    color: var(--text-color-70);
    line-height: 1.35;
    text-decoration: none;
    transition:
      border-color 0.2s ease,
      color 0.2s ease,
      padding-left 0.2s ease;
  }

  .toc-link:hover {
    color: var(--link-color);
    padding-left: 1rem;
  }

  .toc-link.active {
    color: var(--text-color);
    border-left-color: var(--link-color);
    padding-left: 1rem;
  }

  .topic-link {
    display: grid;
    grid-template-columns: 2rem minmax(0, 1fr);
    align-items: start;
    column-gap: 0.65rem;
  }

  .topic-link.current {
    color: var(--text-color);
    border-left-color: var(--link-color);
    padding-left: 0;
  }

  .topic-panel {
    padding: 0;
  }

  .parent-link {
    display: block;
    margin-bottom: 0.75rem;
    padding-left: 0.8rem;
    border-left: 1px solid var(--button-border-color);
    color: var(--text-color-70);
    text-decoration: none;
    transition:
      border-color 0.2s ease,
      color 0.2s ease,
      padding-left 0.2s ease;
  }

  .parent-link:hover {
    color: var(--link-color);
    border-left-color: var(--link-color);
    padding-left: 1rem;
  }

  .parent-kicker {
    display: block;
    margin-bottom: 0.15rem;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-color-70);
  }

  .parent-label {
    display: block;
    font-size: 0.95rem;
    line-height: 1.35;
  }

  .toc-text {
    display: -webkit-box;
    overflow: hidden;
    font-size: 0.9rem;
    line-height: 1.45;
    text-wrap: pretty;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
  }

  .toc-link.active .toc-text {
    font-weight: 600;
  }

  .topic-link.current .toc-text {
    font-weight: 600;
  }

  .topic-link .toc-text {
    font-size: 0.95rem;
    line-height: 1.35;
  }

  .topic-index {
    display: flex;
    justify-content: flex-end;
    width: 2rem;
    padding-top: 0.02rem;
    font-size: 0.95rem;
    line-height: 1.35;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.06em;
    color: var(--text-color-70);
  }

  :global([data-theme="dark"]) .toc-section::before {
    background: color-mix(in srgb, var(--button-border-color) 82%, transparent);
  }

  @media (prefers-reduced-motion: reduce) {
    .toc-link {
      transition: none;
    }

    .parent-link {
      transition: none;
    }
  }
</style>
