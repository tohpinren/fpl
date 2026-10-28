const $ = (selector) => document.querySelector(selector);
const STATIC_MODE = $('meta[name="dashboard-mode"]').content === "static";
if (STATIC_MODE) {
  $("#refresh-button span:last-child").textContent = "Refresh view";
  $("#refresh-button").setAttribute("aria-label", "Refresh view");
  $("#refresh-button").title =
    "Reload the latest published data. Automatic updates run Fridays at 08:17 Singapore time.";
  $("#update-data").hidden = false;
  $("#update-help").hidden = false;
  $("#update-data").title =
    "Open GitHub and choose Run workflow to download and publish fresh data. Repository access required.";
  $("#analysis-source").textContent = "Published FPL analysis";
  $("#method-note").textContent +=
    " Data updates Fridays at 08:17 Singapore time, and on demand via Update data → Run workflow in GitHub. Refresh view reloads the latest published results.";
}
const icons = {
  sliders:
    '<path d="M4 7h16M4 17h16"/><circle cx="8" cy="7" r="2" fill="currentColor" stroke="none"/><circle cx="16" cy="17" r="2" fill="currentColor" stroke="none"/>',
  bookmark: '<path d="M6 4h12v17l-6-4-6 4z"/>',
  calendar:
    '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M16 3v4M8 3v4M3 11h18M8 15h1m6 0h1m-8 3h1"/>',
  "arrow-up-right": '<path d="M7 17 17 7M7 7h10v10"/>',
  "arrow-right": '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  refresh:
    '<path d="M20 7v5h-5M4 17v-5h5M6.3 6.3a8 8 0 0 1 13 2.7M4.7 15a8 8 0 0 0 13 2.7"/>',
  search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 4.5 4.5"/>',
  coins:
    '<circle cx="10" cy="10" r="6"/><path d="M16 8a6 6 0 1 1-8 8M10 7v6m-2-5h3.5a1.5 1.5 0 0 1 0 3H8"/>',
  "chevron-down": '<path d="m7 10 5 5 5-5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6m0-10v.5"/>',
  trend: '<path d="m3 17 6-6 4 4 8-10m-6 0h6v6"/>',
  spark:
    '<path d="m12 3 2.4 6.6L21 12l-6.6 2.4L12 21l-2.4-6.6L3 12l6.6-2.4z"/>',
  pitch:
    '<rect x="5" y="2" width="14" height="20" rx="1"/><path d="M5 12h14M9 2v4h6V2m-6 20v-4h6v4"/><circle cx="12" cy="12" r="3"/>',
};
const icon = (name) =>
  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${icons[name] || icons.info}</svg>`;
document.querySelectorAll("[data-icon]").forEach((element) => {
  element.innerHTML = icon(element.dataset.icon);
});

const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        character
      ],
  );
const money = (value) => `£${Number(value).toFixed(1)}m`;
const decimal = (value, places = 1) => Number(value || 0).toFixed(places);
const positionNames = { DEF: "Defender", MID: "Midfielder", FWD: "Forward" };
const views = {
  targets: "Transfer targets",
  watchlist: "Watchlist",
  fixtures: "Fixture planner",
};
const kitColors = {
  ARS: ["#d94744", "#fff"],
  AVL: ["#805064", "#b8d8e2"],
  BOU: ["#c94d4b", "#282d32"],
  BRE: ["#ca5d53", "#fff"],
  BHA: ["#5e8dc5", "#fff"],
  CHE: ["#5278c8", "#5278c8"],
  COV: ["#92c8db", "#92c8db"],
  CRY: ["#5b78b9", "#c35855"],
  EVE: ["#5472b7", "#5472b7"],
  FUL: ["#e9e8e1", "#373b3c"],
  HUL: ["#e7b553", "#353a39"],
  IPS: ["#587cba", "#587cba"],
  LEE: ["#f1efe6", "#e7d482"],
  LIV: ["#cc5b56", "#cc5b56"],
  MCI: ["#96c2d4", "#96c2d4"],
  MUN: ["#cb5853", "#cb5853"],
  NEW: ["#454c4b", "#f0f0e8"],
  NFO: ["#c75d57", "#c75d57"],
  TOT: ["#f0efe8", "#697184"],
  SUN: ["#cf625b", "#f0eee7"],
};
function shirt(team) {
  const [base, sleeve] = kitColors[team] || ["#819975", "#b1c0a6"];
  return `<span class="team-badge" aria-hidden="true"><svg viewBox="0 0 36 42" fill="none"><path d="m10 5 8-3 8 3 8 8-6 6-3-3v23H11V16l-3 3-6-6z" fill="${base}" stroke="#0000000c" stroke-width=".7"/><path d="m10 5 2 9-4 5-6-6zm16 0-2 9 4 5 6-6z" fill="${sleeve}"/><path d="M13 4q5 9 10 0" stroke="#ffffff99" stroke-width="1.5"/><path d="M11 36h14" stroke="#0000000a"/></svg></span>`;
}
const state = {
  data: null,
  position: "MID",
  view: "targets",
  selectedId: null,
  saved: new Set(),
  visible: [],
  loading: false,
};

function notify(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => {
    $("#toast").hidden = true;
  }, 2500);
}
function status(message, error = false) {
  $("#status-message").textContent = message;
  $("#status-message").hidden = !message;
  $("#status-message").classList.toggle("error", error);
}
function averageDifficulty(player) {
  return player.fixtures.length
    ? player.fixtures.reduce(
        (total, fixture) => total + fixture.difficulty,
        0,
      ) / player.fixtures.length
    : 6;
}
function toggleSaved(id) {
  const player = state.data.players.find((item) => item.id === id);
  if (!player) return;
  const wasSaved = state.saved.has(id);
  if (wasSaved) state.saved.delete(id);
  else state.saved.add(id);
  try {
    localStorage.setItem(
      `touchline-watchlist:${state.data.season}`,
      JSON.stringify([...state.saved]),
    );
  } catch {
    /* Keep the watchlist available for this session. */
  }
  notify(
    `${player.name} ${wasSaved ? "removed from" : "added to"} your watchlist`,
  );
  render();
}
function watchButton(player, detail = false) {
  const isSaved = state.saved.has(player.id);
  return `<button class="watch-button ${isSaved ? "saved" : ""}" data-save="${player.id}" aria-label="${isSaved ? "Remove" : "Save"} ${esc(player.name)} ${isSaved ? "from" : "to"} watchlist" aria-pressed="${isSaved}" ${detail ? "" : 'title="Save to watchlist"'}>${icon("bookmark")}</button>`;
}
function fixtureChip(fixture, showVenue = false) {
  return `<span class="fixture-chip fdr-${fixture.difficulty}" title="GW${fixture.gameweek}: ${esc(fixture.opponent_name)} (${fixture.home ? "home" : "away"}), difficulty ${fixture.difficulty} of 5">${esc(fixture.opponent)}${showVenue ? `<small>${fixture.home ? "HOME" : "AWAY"}</small>` : ""}</span>`;
}
function selectPlayer(id, reveal = false) {
  state.selectedId = id;
  document.querySelectorAll("#player-rows tr").forEach((row) => {
    const selected = Number(row.dataset.player) === id;
    row.classList.toggle("selected", selected);
    row.setAttribute("aria-selected", String(selected));
  });
  renderDetail();
  if (reveal && window.innerWidth <= 820)
    $("#player-detail").scrollIntoView({ behavior: "smooth", block: "start" });
}
function renderInsights() {
  const players = state.data.players.filter(
    (player) => player.position === state.position,
  );
  const leader = players[0];
  const value = [...players]
    .filter((player) => player.status === "a")
    .sort((a, b) => b.points / b.price - a.points / a.price)[0];
  const easy = [...players]
    .filter((player) => player.fixtures.length)
    .sort(
      (a, b) =>
        averageDifficulty(a) - averageDifficulty(b) || b.score - a.score,
    )[0];
  if (!leader) {
    $("#insights").innerHTML = "";
    return;
  }
  const cards = [
    {
      label: `LEADING ${positionNames[state.position].toUpperCase()}`,
      name: leader.name,
      team: leader.team,
      note: `<b>#1</b> in our ${positionNames[state.position].toLowerCase()} rankings`,
      id: leader.id,
      type: "spark",
      featured: true,
    },
    ...(value
      ? [
          {
            label: "VALUE IN THE NUMBERS",
            name: money(value.price),
            note: `<b>${decimal(value.points / value.price)}</b> season points / £m · ${esc(value.name)}`,
            id: value.id,
            type: "trend",
          },
        ]
      : []),
    ...(easy
      ? [
          {
            label: "A KINDER RUN OF FIXTURES",
            name: easy.team_name,
            team: easy.team,
            note: `<b>${decimal(averageDifficulty(easy))}</b> average difficulty · next ${easy.fixtures.length} fixtures`,
            id: easy.id,
            type: "calendar",
          },
        ]
      : []),
  ];
  $("#insights").innerHTML = cards
    .map(
      (card) =>
        `<article class="insight-card ${card.featured ? "featured" : ""}"><div class="insight-label">${card.label}<span class="label-icon">${icon(card.type)}</span></div><div class="insight-main">${card.team ? shirt(card.team) : ""}<strong>${esc(card.name)}</strong></div><div class="insight-note">${card.note}</div><span class="insight-arrow">${icon("arrow-up-right")}</span><button data-insight="${card.id}" aria-label="View ${esc(card.name)}"></button></article>`,
    )
    .join("");
}
function renderDetail() {
  const player = state.data?.players.find(
    (item) => item.id === state.selectedId,
  );
  if (!player) {
    $("#player-detail").innerHTML =
      `<div class="detail-placeholder"><span>${icon("pitch")}</span><p>Select a player to take<br>a closer look.</p></div>`;
    return;
  }
  const available = player.status === "a";
  const availability = available
    ? "Available for selection"
    : player.status === "d"
      ? `Doubtful${player.chance_playing != null ? ` · ${player.chance_playing}% chance of playing` : ""}`
      : player.status === "s"
        ? "Currently suspended"
        : player.status === "i"
          ? "Currently injured"
          : "Currently unavailable";
  const maxPoints = Math.max(10, ...player.recent.map((game) => game.points));
  $("#player-detail").innerHTML = `
    <div class="detail-heading"><div class="detail-topline"><span>PLAYER SPOTLIGHT</span>${watchButton(player, true)}</div><div class="detail-player">${shirt(player.team)}<div><h3>${esc(player.name)}</h3><p>${esc(player.team_name)} · ${positionNames[player.position]}</p></div><span class="detail-price">${money(player.price)}</span></div><div class="availability ${available ? "" : "warning"}"><span></span>${availability}</div>${player.news ? `<p class="news-note">${esc(player.news)}</p>` : ""}</div>
    <div class="detail-stats"><div class="detail-stat"><span>Season points</span><strong>${player.points}<small>pts</small></strong></div><div class="detail-stat"><span>Points per game</span><strong>${decimal(player.points_per_game)}</strong></div><div class="detail-stat"><span>xGI per 90</span><strong>${decimal(player.xgi_per_90, 2)}</strong></div><div class="detail-stat"><span>Selected by</span><strong>${decimal(player.ownership)}<small>%</small></strong></div></div>
    <div class="detail-section"><div class="detail-section-label">Recent returns<span>Last ${player.recent.length} matches</span></div>${player.recent.length ? `<div class="form-chart" role="img" aria-label="${esc(player.recent.map((game) => `Gameweek ${game.gameweek}: ${game.points} points`).join(", "))}">${player.recent.map((game) => `<div class="form-bar-wrap"><span class="bar-value">${game.points}</span><div class="form-bar" style="height:${Math.max(4, (game.points / maxPoints) * 64)}px"></div><span class="bar-label">GW${game.gameweek}</span></div>`).join("")}</div>` : '<p class="planner-note">No completed matches yet.</p>'}</div>
    <div class="detail-section"><div class="detail-section-label">The road ahead<span>FPL difficulty</span></div><div class="detail-fixtures">${player.fixtures.length ? player.fixtures.map((fixture) => `<div class="detail-fixture"><span class="fixture-gw">GW${fixture.gameweek}</span><span class="opponent-name">${esc(fixture.opponent_name)}</span><span class="home-away">${fixture.home ? "H" : "A"}</span><span class="difficulty-number fdr-${fixture.difficulty}">${fixture.difficulty}</span></div>`).join("") : '<p class="planner-note">No upcoming fixtures scheduled.</p>'}</div></div>`;
}
function filteredPlayers() {
  const query = $("#search").value.trim().toLowerCase();
  const price = Number($("#price-filter").value);
  const sort = $("#sort-select").value;
  const players = state.data.players.filter(
    (player) =>
      player.position === state.position &&
      player.price <= price &&
      `${player.name} ${player.full_name} ${player.team} ${player.team_name}`
        .toLowerCase()
        .includes(query) &&
      (state.view !== "watchlist" || state.saved.has(player.id)),
  );
  return players.sort((a, b) =>
    sort === "price"
      ? a.price - b.price || b.score - a.score
      : sort === "fixtures"
        ? averageDifficulty(a) - averageDifficulty(b) || b.score - a.score
        : b[sort] - a[sort],
  );
}
function renderPlanner() {
  const query = $("#search").value.trim().toLowerCase();
  const teamMap = new Map();
  state.data.players.forEach((player) => {
    if (!teamMap.has(player.team)) teamMap.set(player.team, player);
  });
  let teams = [...teamMap.values()].filter((player) =>
    `${player.team_name} ${player.team}`.toLowerCase().includes(query),
  );
  teams.sort(
    (a, b) =>
      averageDifficulty(a) - averageDifficulty(b) ||
      a.team_name.localeCompare(b.team_name),
  );
  $("#players-table").classList.add("fixture-planner");
  const gameweeks = [
    ...new Set(
      teams.flatMap((team) => team.fixtures.map((fixture) => fixture.gameweek)),
    ),
  ]
    .sort((a, b) => a - b)
    .slice(0, 5);
  $("#players-table thead").innerHTML =
    `<tr><th class="rank-cell">#</th><th>Club</th>${gameweeks.map((gw) => `<th>GW${gw}</th>`).join("")}<th class="numeric">Avg. FDR</th></tr>`;
  $("#player-rows").innerHTML = teams
    .map(
      (team, index) =>
        `<tr data-player="${team.id}" tabindex="0" aria-label="View ${esc(team.team_name)} fixtures"><td class="rank-cell">${index + 1}</td><td><div class="player-cell">${shirt(team.team)}<div class="planner-team">${esc(team.team_name)}<div class="player-subtitle">${esc(team.team)}</div></div></div></td>${gameweeks
          .map(
            (gw) =>
              `<td>${
                team.fixtures
                  .filter((fixture) => fixture.gameweek === gw)
                  .map((fixture) => fixtureChip(fixture, true))
                  .join(" ") || '<span class="planner-empty">—</span>'
              }</td>`,
          )
          .join(
            "",
          )}<td class="numeric score-cell">${averageDifficulty(team) <= 5 ? decimal(averageDifficulty(team)) : "—"}</td></tr>`,
    )
    .join("");
  $("#empty-state").hidden = teams.length > 0;
  $("#results-count").textContent =
    `${teams.length} clubs · ranked by upcoming fixture difficulty`;
  state.visible = teams;
}
const originalTableHead = $("#players-table thead").innerHTML;
function render() {
  if (!state.data) return;
  $("#watchlist-count").textContent = state.data.players.filter((player) =>
    state.saved.has(player.id),
  ).length;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
    if (button.dataset.view === state.view)
      button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  $("#breadcrumb-view").textContent = views[state.view];
  document.querySelectorAll("[data-position]").forEach((button) => {
    const active = button.dataset.position === state.position;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    $(`#count-${button.dataset.position}`).textContent =
      state.data.players.filter(
        (player) =>
          player.position === button.dataset.position &&
          (state.view !== "watchlist" || state.saved.has(player.id)),
      ).length;
  });
  const planner = state.view === "fixtures";
  $(".position-bar").hidden = planner;
  $(".filter-selects").hidden = planner;
  $("#search").placeholder = planner ? "Find a club" : "Find a player or club";
  $("#section-title").textContent = planner
    ? "Find the favourable fixtures"
    : state.view === "watchlist"
      ? "Your watchlist"
      : "The shortlist";
  $("#section-description").textContent = planner
    ? "A little forward planning goes a long way."
    : state.view === "watchlist"
      ? "The players you’re keeping an eye on. Saved on this browser."
      : "Good players. Useful numbers. A clearer decision.";
  renderInsights();
  if (planner) renderPlanner();
  else {
    $("#players-table").classList.remove("fixture-planner");
    $("#players-table thead").innerHTML = originalTableHead;
    const players = filteredPlayers();
    state.visible = players;
    $("#player-rows").innerHTML = players
      .map(
        (player) =>
          `<tr data-player="${player.id}" tabindex="0" class="${player.id === state.selectedId ? "selected" : ""}" aria-selected="${player.id === state.selectedId}" aria-label="View ${esc(player.name)} details"><td class="rank-cell">${player.rank}</td><td><div class="player-cell">${shirt(player.team)}<div><div class="player-name">${esc(player.name)}${player.status !== "a" ? '<span class="status-dot" title="Availability concern"></span>' : ""}</div><div class="player-subtitle">${esc(player.team)}<span class="position-mini">${player.position}</span></div></div></div></td><td class="numeric price-cell">${money(player.price)}</td><td class="numeric ${player.form >= 5 ? "form-good" : "form-normal"}">${decimal(player.form)}</td><td class="numeric xgi-column form-normal">${decimal(player.xgi_per_90, 2)}</td><td class="fixtures-column"><div class="fixture-list">${player.fixtures.map((fixture) => fixtureChip(fixture)).join("") || '<span class="planner-empty">—</span>'}</div></td><td class="numeric score-cell"><span>${decimal(player.score)}</span></td><td>${watchButton(player)}</td></tr>`,
      )
      .join("");
    $("#empty-state").hidden = players.length > 0;
    if (!players.length) {
      $("#empty-state h3").textContent =
        state.view === "watchlist"
          ? "A little room for potential"
          : "No players in view";
      $("#empty-state p").textContent =
        state.view === "watchlist"
          ? "Save players with the bookmark icon, or try another position."
          : "Try another position or loosen your filters.";
    }
    $("#results-count").textContent =
      `${players.length} ${positionNames[state.position].toLowerCase()}${players.length !== 1 ? "s" : ""} · 180+ minutes played`;
  }
  if (!state.visible.some((player) => player.id === state.selectedId))
    state.selectedId = state.visible[0]?.id ?? null;
  selectPlayer(state.selectedId);
}
function acceptData(data) {
  if (state.data?.season !== data.season) {
    try {
      state.saved = new Set(
        JSON.parse(
          localStorage.getItem(`touchline-watchlist:${data.season}`) || "[]",
        ).filter(Number.isInteger),
      );
    } catch {
      state.saved = new Set();
    }
  }
  state.data = data;
  $("#reset-filters").hidden = false;
  $("#season-tag").textContent = `SEASON ${data.season}`;
  $("#gw-label").textContent = data.gameweek
    ? `Gameweek ${data.gameweek}`
    : "Season complete";
  $("#deadline").textContent = data.deadline
    ? new Intl.DateTimeFormat("en-GB", {
        weekday: "short",
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
        timeZoneName: "short",
      }).format(new Date(data.deadline))
    : "No upcoming deadline";
  $("#deadline").title = "Deadline shown in your local timezone";
  const updated = new Date(data.updated_at);
  $("#updated-label").textContent =
    `Updated ${new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }).format(updated)}`;
  $("#updated-label").title = updated.toLocaleString();
  render();
}
async function load(refresh = false) {
  if (state.loading) return;
  state.loading = true;
  const button = $("#refresh-button");
  button.disabled = true;
  button.classList.add("busy");
  if (refresh)
    status(
      STATIC_MODE
        ? "Loading the latest published results…"
        : "Refreshing current-season FPL data. This can take a moment…",
    );
  try {
    const response = await fetch(
      STATIC_MODE
        ? "./dashboard.json"
        : refresh
          ? "./api/refresh"
          : "./api/dashboard",
      refresh && !STATIC_MODE ? { method: "POST" } : { cache: "no-store" },
    );
    if (!response.ok && STATIC_MODE)
      throw new Error(
        "Published data is temporarily unavailable. Please try Refresh view shortly.",
      );
    const data = await response.json();
    if (!response.ok)
      throw new Error(data.error || "Could not load the dashboard.");
    acceptData(data);
    status("");
    if (refresh)
      notify(
        STATIC_MODE
          ? "Showing the latest published results"
          : "Your FPL data is up to date",
      );
  } catch (error) {
    status(error.message || "Something went wrong. Please try again.", true);
    if (!state.data) {
      $("#insights").innerHTML = "";
      $("#results-count").textContent = "No data loaded";
      $("#updated-label").textContent = "Data unavailable";
      $("#empty-state").hidden = false;
      $("#empty-state h3").textContent = STATIC_MODE
        ? "We couldn’t load the latest results"
        : "Let’s get your data ready";
      $("#empty-state p").textContent = STATIC_MODE
        ? "Use Refresh view to try loading the published data again."
        : "Use Refresh data to download the current FPL season.";
      $("#reset-filters").hidden = true;
    }
  } finally {
    state.loading = false;
    button.disabled = false;
    button.classList.remove("busy");
  }
}
document.addEventListener("click", (event) => {
  const save = event.target.closest("[data-save]");
  if (save) {
    toggleSaved(Number(save.dataset.save));
    return;
  }
  const row = event.target.closest("[data-player]");
  if (row) {
    selectPlayer(Number(row.dataset.player), true);
    return;
  }
  const insight = event.target.closest("[data-insight]");
  if (insight) {
    const id = Number(insight.dataset.insight);
    if (state.view !== "targets") setView("targets");
    $("#search").value = "";
    $("#price-filter").value = "99";
    render();
    selectPlayer(id, true);
    document
      .querySelector(`[data-player="${id}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
});
function setView(view) {
  state.view = view;
  $("#search").value = "";
  render();
}
document
  .querySelectorAll("[data-view]")
  .forEach((button) =>
    button.addEventListener("click", () => setView(button.dataset.view)),
  );
document.querySelectorAll("[data-position]").forEach((button) =>
  button.addEventListener("click", () => {
    state.position = button.dataset.position;
    render();
  }),
);
$(".position-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
  const tabs = [...document.querySelectorAll("[data-position]")];
  const current = tabs.findIndex(
    (tab) => tab.dataset.position === state.position,
  );
  const next =
    event.key === "Home"
      ? 0
      : event.key === "End"
        ? tabs.length - 1
        : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) %
          tabs.length;
  tabs[next].click();
  tabs[next].focus();
  event.preventDefault();
});
$("#search").addEventListener("input", render);
$("#price-filter").addEventListener("change", render);
$("#sort-select").addEventListener("change", render);
$("#refresh-button").addEventListener("click", () => load(true));
$("#method-button").addEventListener("click", () => {
  $("#method-note").hidden = !$("#method-note").hidden;
  $("#method-button").setAttribute(
    "aria-expanded",
    String(!$("#method-note").hidden),
  );
});
$("#method-button").setAttribute("aria-expanded", "false");
$("#method-button").setAttribute("aria-controls", "method-note");
$("#reset-filters").addEventListener("click", () => {
  $("#search").value = "";
  $("#price-filter").value = "99";
  $("#sort-select").value = "score";
  render();
});
document.addEventListener("keydown", (event) => {
  if (
    event.key === "/" &&
    !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)
  ) {
    event.preventDefault();
    $("#search").focus();
  }
  const row = event.target.closest("[data-player]");
  if (row && event.target === row && ["Enter", " "].includes(event.key)) {
    event.preventDefault();
    selectPlayer(Number(row.dataset.player), true);
  }
});
load();
