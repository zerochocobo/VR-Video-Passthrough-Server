(() => {
  const languages = {
    en: { tag: "en", title: "Privacy Policy", home: "← Home", description: "Privacy policy for Thru3D Media Player on Meta Quest and PICO, and Thru3D Media Server for Windows." },
    zh: { tag: "zh-CN", title: "隐私政策", home: "← 首页", description: "适用于 Meta Quest、PICO 版 Thru3D Media Player 及 Windows 版 Thru3D媒体服务器 的隐私政策。" },
    ja: { tag: "ja", title: "プライバシーポリシー", home: "← ホーム", description: "Meta Quest・PICO 向け Thru3D Media Player と Windows 向け Thru3D Media Server のプライバシーポリシー。" },
  };
  const normalize = (value) => {
    const primary = String(value || "").toLowerCase().split(/[-_]/)[0];
    return Object.prototype.hasOwnProperty.call(languages, primary) ? primary : null;
  };
  const hashLanguage = () => normalize(location.hash.slice(1));
  const savedLanguage = () => {
    try { return normalize(localStorage.getItem("ptserver-language")); }
    catch { return null; }
  };
  const applyLanguage = (language) => {
    const copy = languages[language];
    document.documentElement.lang = copy.tag;
    const serverName = language === "zh" ? "Thru3D媒体服务器" : "Thru3D Media Server";
    document.title = `${copy.title} | Thru3D Media Player & ${serverName}`;
    document.querySelector("[data-server-name]").textContent = serverName;
    document.querySelector('meta[name="description"]').content = copy.description;
    document.querySelectorAll("[data-policy]").forEach((article) => {
      article.hidden = article.dataset.policy !== language;
    });
    document.querySelectorAll("[data-policy-language]").forEach((link) => {
      link.setAttribute("aria-current", String(link.dataset.policyLanguage === language));
    });
    document.querySelectorAll("[data-home-link]").forEach((link) => {
      link.href = `index.html?lang=${language}`;
    });
    document.querySelector("[data-home-label]").textContent = copy.home;
    try { localStorage.setItem("ptserver-language", language); } catch { /* Storage is optional. */ }
  };
  const fromUrl = () => hashLanguage() || normalize(new URLSearchParams(location.search).get("lang"));
  const initial = fromUrl() || savedLanguage() || normalize(navigator.language) || "en";
  applyLanguage(initial);
  document.querySelectorAll("[data-policy-language]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const language = link.dataset.policyLanguage;
      const url = new URL(location.href);
      url.searchParams.set("lang", language);
      url.hash = language;
      history.pushState(null, "", url);
      applyLanguage(language);
      window.scrollTo({ top: 0, behavior: "instant" });
    });
  });
  window.addEventListener("popstate", () => applyLanguage(fromUrl() || initial));
  window.addEventListener("hashchange", () => {
    const language = hashLanguage();
    if (language) applyLanguage(language);
  });
})();
