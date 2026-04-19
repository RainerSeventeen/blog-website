<script>
  import { onMount } from "svelte";

  let { headings = [] } = $props();

  let activeSlug = $state("");
  let tocListElement = $state(null);
  let headingElements = $state([]);
  let ticking = false;

  const filteredHeadings = $derived(headings.filter((heading) => heading.depth === 2));

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

    if (!headingElements.length) {
      return;
    }

    requestHeadingSync();

    window.addEventListener("scroll", requestHeadingSync, { passive: true });
    window.addEventListener("resize", requestHeadingSync);

    return () => {
      window.removeEventListener("scroll", requestHeadingSync);
      window.removeEventListener("resize", requestHeadingSync);
    };
  });
</script>

{#if filteredHeadings.length > 0}
  <aside class="toc-sidebar" aria-labelledby="toc-heading">
    <div class="toc-card">
      <div class="toc-heading">
        <h2 id="toc-heading" class="toc-title">目录</h2>
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
      left: calc(50vw + var(--page-width) / 2 + 2rem);
      width: min(var(--toc-panel-width), calc(100vw - (50vw + var(--page-width) / 2) - 3rem));
      display: block;
    }
  }

  .toc-card {
    border: 1px solid var(--button-border-color);
    border-radius: 1rem;
    background: rgba(251, 251, 251, 0.88);
    backdrop-filter: blur(14px);
    box-shadow: 0 24px 48px -36px var(--shadow-color);
    overflow: hidden;
  }

  .toc-heading {
    padding: 0.85rem 1rem 0.65rem;
    border-bottom: 1px solid var(--button-border-color);
  }

  .toc-title {
    margin: 0;
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: var(--text-color-70);
  }

  .toc-list {
    max-height: min(70vh, 38rem);
    margin: 0;
    padding: 0.45rem 0.5rem;
    list-style: none;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--scrollbar-thumb) transparent;
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
    padding: 0.38rem 0.55rem;
    border-radius: 0.65rem;
    color: var(--text-color-70);
    line-height: 1.35;
    text-decoration: none;
    transition:
      background-color 0.2s ease,
      color 0.2s ease;
  }

  .toc-link:hover {
    background: var(--button-hover-color);
    color: var(--text-color);
  }

  .toc-link.active {
    background: var(--button-hover-color);
    color: var(--text-color);
  }

  .toc-text {
    display: -webkit-box;
    overflow: hidden;
    font-size: 0.9rem;
    line-height: 1.35;
    text-wrap: pretty;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
  }

  .toc-link.active .toc-text {
    font-weight: 600;
  }

  :global([data-theme="dark"]) .toc-card {
    background: rgba(10, 10, 10, 0.86);
  }

  @media (prefers-reduced-motion: reduce) {
    .toc-link {
      transition: none;
    }
  }
</style>
