document.addEventListener("DOMContentLoaded", () => {
  const burger = document.querySelector(".navbar-burger");
  const menu = document.getElementById("main-nav");
  if (burger && menu) {
    burger.addEventListener("click", () => {
      const active = burger.classList.toggle("is-active");
      menu.classList.toggle("is-active", active);
      burger.setAttribute("aria-expanded", String(active));
    });
    menu.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => {
      burger.classList.remove("is-active");
      menu.classList.remove("is-active");
      burger.setAttribute("aria-expanded", "false");
    }));
  }

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.innerText);
        const original = button.textContent;
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = original; }, 1600);
      } catch (_) {
        button.textContent = "Select text to copy";
      }
    });
  });
});
