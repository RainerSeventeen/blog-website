<script>
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import { spring } from "svelte/motion";
  import { siteConfig } from "@/config";

  let { headings = [] } = $props();

  let tocVisible = $state(false);
  let activeIndex = $state(-1);
  let tocListElement = $state(null);

  const focusSpring = spring(-1, {
    stiffness: 0.12,
    damping: 0.7,
  });

  let minDepth = $derived(headings.length > 0 ? Math.min(...headings.map((heading) => heading.depth)) : 0);
  let maxDepth = $derived(minDepth + (siteConfig?.toc?.depth || 2));
  let filteredHeadings = $derived(headings.filter((heading) => heading.depth < maxDepth));

  $effect(() => {
    focusSpring.set(activeIndex);
  });

  function handleScroll() {
    tocVisible = (window.scrollY || window.pageYOffset) > 200;
  }

  function initObserver() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const index = filteredHeadings.findIndex((heading) => heading.slug === entry.target.id);
            if (index !== -1) {
              activeIndex = index;
              autoScrollTOC(index);
            }
          }
        });
      },
      { rootMargin: "-10% 0px -70% 0px", threshold: 0.1 },
    );

    filteredHeadings.forEach((heading) => {
      const element = document.getElementById(heading.slug);
      if (element) {
        observer.observe(element);
      }
    });

    return observer;
  }

  function autoScrollTOC(index) {
    if (!tocListElement) {
      return;
    }

    const items = tocListElement.querySelectorAll("a");
    const activeItem = items[index];
    if (!activeItem) {
      return;
    }

    const containerHeight = tocListElement.clientHeight;
    const targetScroll = activeItem.offsetTop - containerHeight / 2 + activeItem.clientHeight / 2;
    tocListElement.scrollTo({ top: targetScroll, behavior: "smooth" });
  }

  function getSpringStyle(index, currentSpring) {
    const distance = Math.abs(index - currentSpring);
    const opacity = Math.max(0.2, 1 - distance * 0.2);
    const fontWeight = distance < 0.5 ? "700" : "400";
    return `opacity: ${opacity}; font-weight: ${fontWeight};`;
  }

  onMount(() => {
    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    const observer = initObserver();

    return () => {
      window.removeEventListener("scroll", handleScroll);
      observer.disconnect();
    };
  });
</script>

{#if tocVisible}
  <aside transition:fade={{ duration: 300 }} class="fixed left-[var(--toc-offset-left)] top-20 z-10 hidden w-[var(--toc-width)] text-[var(--text-color)] lg:block">
    <div class="flex h-[50vh] flex-col bg-transparent">
      <h2 id="toc-heading" class="mb-2 text-lg font-bold uppercase tracking-widest">
        目录
      </h2>

      <ul bind:this={tocListElement} class="no-scrollbar space-y-2 overflow-y-auto pr-4" style="scrollbar-width: none; scroll-behavior: smooth;">
        {#each filteredHeadings as heading, index}
          <li>
            <a
              href={`#${heading.slug}`}
              class="block py-1 text-sm transition-colors duration-300 hover:text-[var(--link-color)]"
              style:padding-left="{(heading.depth - minDepth) * 1.2}rem"
              style={getSpringStyle(index, $focusSpring)}
              onclick={(event) => {
                event.preventDefault();
                document.getElementById(heading.slug)?.scrollIntoView({ behavior: "smooth" });
              }}
            >
              {heading.text}
            </a>
          </li>
        {/each}
      </ul>
    </div>
  </aside>
{/if}

<style>
  .no-scrollbar::-webkit-scrollbar {
    display: none;
  }

  a {
    display: block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    will-change: opacity, font-weight;
  }
</style>
