const DATASETS = {
  training: { label: "Training", path: "/data/training/", count: 1000 },
  evaluation: { label: "Public eval", path: "/data/evaluation/", count: 120 },
};

const PALETTE_NAMES = ["black", "red", "orange", "yellow", "green", "blue", "dark blue", "purple", "light gray", "gray"];
const state = {
  dataset: "training",
  files: [],
  filtered: [],
  index: 0,
  task: null,
  taskId: "",
  challenge: false,
  revealedAnswers: new Set(),
  guesses: new Map(),
  submissions: new Map(),
};

const $ = (selector) => document.querySelector(selector);
const els = {
  loading: $("#loading-state"),
  explorer: $("#explorer-layout"),
  error: $("#error-state"),
  taskList: $("#task-list"),
  search: $("#task-search"),
  taskPosition: $("#task-position"),
  queueTitle: $("#queue-title"),
  queueCount: $("#queue-count"),
  overviewName: $("#overview-name"),
  overviewDetail: $("#overview-detail"),
  overviewBar: $("#overview-bar-fill"),
  stageDataset: $("#stage-dataset"),
  stageIndex: $("#stage-index"),
  taskTitle: $("#task-title"),
  taskMeta: $("#task-meta"),
  taskNotice: $("#task-notice"),
  trainCount: $("#train-count"),
  testCount: $("#test-count"),
  testNoteCopy: $("#test-note-copy"),
  trainPairs: $("#train-pairs"),
  testPairs: $("#test-pairs"),
  challengeToggle: $("#challenge-toggle"),
  challengeState: $("#challenge-state"),
  focusModal: $("#focus-modal"),
  focusTitle: $("#focus-title"),
  focusMeta: $("#focus-meta"),
  focusCanvas: $("#focus-canvas"),
};

async function discoverFiles(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not read ${path}`);
  const html = await response.text();
  const parsed = new DOMParser().parseFromString(html, "text/html");
  const links = [...parsed.querySelectorAll("a")].map((link) => link.getAttribute("href") || "");
  const files = links
    .map((href) => decodeURIComponent(href.split("/").pop() || ""))
    .filter((name) => /^[a-z0-9]+\.json$/i.test(name));
  if (!files.length) throw new Error("No JSON task files found");
  return files.sort((a, b) => a.localeCompare(b));
}

async function loadDataset(dataset) {
  state.dataset = dataset;
  state.task = null;
  state.taskId = "";
  const config = DATASETS[dataset];
  state.files = await discoverFiles(config.path);
  state.filtered = [...state.files];
  state.index = 0;
  setDatasetTab();
  renderQueue();
  await loadCurrentTask();
}

function setDatasetTab() {
  document.querySelectorAll(".dataset-tab").forEach((tab) => {
    const active = tab.dataset.dataset === state.dataset;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  const config = DATASETS[state.dataset];
  els.queueTitle.textContent = `${config.label} tasks`;
  els.overviewName.textContent = config.label;
  els.overviewDetail.textContent = `${state.files.length.toLocaleString()} local task files`;
  els.overviewBar.style.width = `${state.dataset === "training" ? 88 : 100}%`;
  const savedChallenge = localStorage.getItem("arc-explorer-challenge");
  state.challenge = savedChallenge === null ? state.dataset === "evaluation" : savedChallenge === "true";
  updateChallengeToggle();
}

function updateChallengeToggle() {
  const active = state.challenge;
  els.challengeToggle.classList.toggle("is-on", active);
  els.challengeToggle.setAttribute("aria-pressed", String(active));
  els.challengeState.textContent = active ? "on" : "off";
  els.testNoteCopy.textContent = active
    ? "Challenge mode is on. Paint your guess, check it, and reveal only when you are ready."
    : "Expected outputs are visible because this is the local public dataset.";
}

function renderQueue() {
  const fragment = document.createDocumentFragment();
  const selectedId = state.filtered[state.index];
  els.queueCount.textContent = state.filtered.length.toLocaleString();
  els.taskPosition.textContent = state.filtered.length ? `${String(state.index + 1).padStart(3, "0")} / ${String(state.filtered.length).padStart(3, "0")}` : "— / —";
  if (!state.filtered.length) {
    const empty = document.createElement("div");
    empty.className = "empty-queue";
    empty.textContent = "No task IDs match that search.";
    els.taskList.replaceChildren(empty);
    return;
  }
  state.filtered.forEach((file, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `task-item${file === selectedId ? " is-selected" : ""}`;
    button.dataset.file = file;
    button.innerHTML = `<span class="task-item-id">${file.replace(".json", "")}</span><span class="task-item-index">${String(index + 1).padStart(3, "0")}</span>`;
    button.addEventListener("click", () => selectTask(file));
    fragment.append(button);
  });
  els.taskList.replaceChildren(fragment);
  const selected = els.taskList.querySelector(".is-selected");
  selected?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function selectTask(file) {
  const nextIndex = state.filtered.indexOf(file);
  if (nextIndex < 0) return;
  state.index = nextIndex;
  closeFocusView();
  renderQueue();
  loadCurrentTask();
}

async function loadCurrentTask() {
  if (!state.filtered.length) {
    els.taskTitle.textContent = "No matching tasks";
    els.taskMeta.textContent = "Try a different task ID.";
    els.trainPairs.replaceChildren();
    els.testPairs.replaceChildren();
    return;
  }
  const id = state.filtered[state.index];
  state.taskId = id.replace(/\.json$/i, "");
  els.taskTitle.textContent = state.taskId;
  els.stageDataset.textContent = DATASETS[state.dataset].label.toUpperCase();
  els.stageIndex.textContent = `TASK ${String(state.index + 1).padStart(3, "0")}`;
  els.taskMeta.textContent = "Loading grid details…";
  els.taskNotice.hidden = true;
  try {
    const response = await fetch(`${DATASETS[state.dataset].path}${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error("Task request failed");
    state.task = await response.json();
    renderTask();
  } catch (error) {
    state.task = null;
    els.taskNotice.hidden = false;
    els.taskMeta.textContent = "Could not load this task";
    els.trainPairs.replaceChildren();
    els.testPairs.replaceChildren();
    console.error(error);
  }
}

function renderTask() {
  const task = state.task;
  const train = task.train || [];
  const test = task.test || [];
  const visibleGrids = [
    ...train.flatMap((pair) => [pair.input, pair.output]),
    ...test.flatMap((pair) => state.challenge ? [pair.input] : [pair.input, pair.output]),
  ].filter(Boolean);
  const maxRows = visibleGrids.reduce((largest, grid) => Math.max(largest, grid.length), 0);
  const maxCols = visibleGrids.reduce((largest, grid) => Math.max(largest, grid[0]?.length || 0), 0);
  const colors = [...new Set(visibleGrids.flat(2))].sort((a, b) => a - b);
  els.taskMeta.textContent = `${train.length} demonstration${train.length === 1 ? "" : "s"} · ${test.length} test input${test.length === 1 ? "" : "s"} · largest grid ${maxRows} × ${maxCols} · colors ${colors.join(", ")}`;
  els.trainCount.textContent = `${train.length} demonstration${train.length === 1 ? "" : "s"}`;
  els.testCount.textContent = `${test.length} test input${test.length === 1 ? "" : "s"}`;
  updateChallengeToggle();
  els.trainPairs.replaceChildren(...train.map((pair, index) => renderPair(pair, index, "demonstration")));
  els.testPairs.replaceChildren(...test.map((pair, index) => renderPair(pair, index, "test")));
}

function renderPair(pair, index, kind) {
  const row = document.createElement("article");
  row.className = "pair-row";
  const inputLabel = kind === "test" ? `Test input ${index + 1}` : `Example ${index + 1} · input`;
  const outputLabel = kind === "test" ? `Expected output ${index + 1}` : `Example ${index + 1} · output`;
  const answerKey = `${state.dataset}:${state.taskId}:${index}`;
  const hiddenAnswer = kind === "test" && state.challenge && !state.revealedAnswers.has(answerKey);
  const output = hiddenAnswer
    ? renderAnswerEditor(pair.input, pair.output, answerKey)
    : renderGridCard(outputLabel, pair.output);
  row.append(renderGridCard(inputLabel, pair.input), renderArrow(), output);
  return row;
}

function renderGridCard(label, grid, options = {}) {
  const figure = document.createElement("figure");
  figure.className = "grid-card";
  const rows = grid?.length || 0;
  const cols = grid?.[0]?.length || 0;
  const head = document.createElement("figcaption");
  head.className = "grid-card-head";
  const title = document.createElement("strong");
  title.textContent = label;
  const dimensions = document.createElement("span");
  dimensions.textContent = `${rows} × ${cols}`;
  head.append(title, dimensions);
  if (!options.hiddenAnswer) {
    const focusButton = document.createElement("button");
    focusButton.type = "button";
    focusButton.className = "grid-focus";
    focusButton.textContent = "Focus view";
    focusButton.setAttribute("aria-label", `Open ${label} in focus view`);
    focusButton.addEventListener("click", () => openFocusView(label, grid));
    head.append(focusButton);
  }
  const frame = document.createElement("div");
  if (options.hiddenAnswer) {
    frame.className = "answer-cover";
    frame.innerHTML = `<strong>Answer hidden</strong><p>Form your best answer before checking the expected grid.</p>`;
    const revealButton = document.createElement("button");
    revealButton.type = "button";
    revealButton.className = "reveal-button";
    revealButton.textContent = "Reveal answer";
    revealButton.addEventListener("click", () => revealAnswer(options.answerKey));
    frame.append(revealButton);
  } else {
    frame.className = "grid-frame";
    frame.append(renderGridElement(grid));
  }
  figure.append(head, frame);
  return figure;
}

function renderGridElement(grid, focus = false) {
  const rows = grid?.length || 0;
  const cols = grid?.[0]?.length || 0;
  const gridElement = document.createElement("div");
  gridElement.className = `arc-grid${focus ? " focus-grid" : ""}`;
  gridElement.style.setProperty("--cols", cols);
  gridElement.style.setProperty("--rows", rows);
  gridElement.setAttribute("role", "img");
  gridElement.setAttribute("aria-label", `${rows} by ${cols} grid`);
  grid.flat().forEach((value) => {
    const cell = document.createElement("span");
    cell.className = `cell cell-${Number(value)}`;
    cell.title = `${PALETTE_NAMES[Number(value)] || "color"} · ${value}`;
    gridElement.append(cell);
  });
  return gridElement;
}

function emptyGrid(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function getGuess(answerKey, inputGrid) {
  const rows = Math.max(1, inputGrid?.length || 1);
  const cols = Math.max(1, inputGrid?.[0]?.length || 1);
  let guess = state.guesses.get(answerKey);
  if (!guess) {
    guess = { rows, cols, color: 0, cells: emptyGrid(rows, cols) };
    state.guesses.set(answerKey, guess);
  }
  return guess;
}

function resizeGuess(guess, rows, cols) {
  const next = emptyGrid(rows, cols);
  for (let row = 0; row < Math.min(rows, guess.rows); row += 1) {
    for (let col = 0; col < Math.min(cols, guess.cols); col += 1) next[row][col] = guess.cells[row][col];
  }
  guess.rows = rows;
  guess.cols = cols;
  guess.cells = next;
}

function createSizeControl(label, key, guess, answerKey) {
  const control = document.createElement("div");
  control.className = "size-control";
  const labelElement = document.createElement("span");
  labelElement.textContent = label;
  const stepper = document.createElement("div");
  stepper.className = "size-stepper";
  const minus = document.createElement("button");
  minus.type = "button";
  minus.className = "size-button";
  minus.textContent = "−";
  minus.setAttribute("aria-label", `Decrease ${label.toLowerCase()}`);
  const value = document.createElement("output");
  value.textContent = guess[key];
  value.setAttribute("aria-label", `${label}: ${guess[key]}`);
  const plus = document.createElement("button");
  plus.type = "button";
  plus.className = "size-button";
  plus.textContent = "+";
  plus.setAttribute("aria-label", `Increase ${label.toLowerCase()}`);
  const change = (delta) => {
    const next = Math.max(1, Math.min(30, guess[key] + delta));
    if (next === guess[key]) return;
    const rows = key === "rows" ? next : guess.rows;
    const cols = key === "cols" ? next : guess.cols;
    resizeGuess(guess, rows, cols);
    state.submissions.delete(answerKey);
    renderTask();
  };
  minus.addEventListener("click", () => change(-1));
  plus.addEventListener("click", () => change(1));
  stepper.append(minus, value, plus);
  control.append(labelElement, stepper);
  return control;
}

function isExactMatch(guess, expected) {
  const rows = expected?.length || 0;
  const cols = expected?.[0]?.length || 0;
  if (guess.rows !== rows || guess.cols !== cols) return false;
  return guess.cells.every((row, rowIndex) => row.every((value, colIndex) => value === expected[rowIndex][colIndex]));
}

function renderAnswerEditor(inputGrid, expectedGrid, answerKey) {
  const guess = getGuess(answerKey, inputGrid);
  const figure = document.createElement("figure");
  figure.className = "grid-card answer-builder-card";

  const head = document.createElement("figcaption");
  head.className = "grid-card-head";
  const title = document.createElement("strong");
  title.textContent = "Your answer";
  const dimensions = document.createElement("span");
  dimensions.textContent = `${guess.rows} × ${guess.cols}`;
  head.append(title, dimensions);

  const builder = document.createElement("div");
  builder.className = "answer-builder";

  const sizeControls = document.createElement("div");
  sizeControls.className = "builder-toolbar";
  sizeControls.append(createSizeControl("Rows", "rows", guess, answerKey), createSizeControl("Columns", "cols", guess, answerKey));
  const helper = document.createElement("span");
  helper.className = "builder-helper";
  helper.textContent = "Click a cell to paint it";
  sizeControls.append(helper);
  builder.append(sizeControls);

  const palette = document.createElement("div");
  palette.className = "answer-palette";
  PALETTE_NAMES.forEach((name, value) => {
    const swatch = document.createElement("button");
    swatch.type = "button";
    swatch.className = `palette-swatch cell-${value}${guess.color === value ? " is-selected" : ""}`;
    swatch.textContent = value;
    swatch.title = `${name} · ${value}`;
    swatch.setAttribute("aria-label", `Choose ${name}, color ${value}`);
    swatch.setAttribute("aria-pressed", String(guess.color === value));
    swatch.addEventListener("click", () => {
      guess.color = value;
      palette.querySelectorAll(".palette-swatch").forEach((item) => {
        const selected = item === swatch;
        item.classList.toggle("is-selected", selected);
        item.setAttribute("aria-pressed", String(selected));
      });
    });
    palette.append(swatch);
  });
  builder.append(palette);

  const frame = document.createElement("div");
  frame.className = "editor-frame";
  const editorGrid = document.createElement("div");
  editorGrid.className = "arc-grid editor-grid";
  editorGrid.style.setProperty("--cols", guess.cols);
  editorGrid.style.setProperty("--rows", guess.rows);
  editorGrid.setAttribute("role", "grid");
  editorGrid.setAttribute("aria-label", `Your answer, ${guess.rows} by ${guess.cols} grid`);
  guess.cells.forEach((row, rowIndex) => row.forEach((value, colIndex) => {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = `cell editor-cell cell-${Number(value)}`;
    cell.title = `${PALETTE_NAMES[Number(value)] || "color"} · ${value}`;
    cell.setAttribute("role", "gridcell");
    cell.setAttribute("aria-label", `Row ${rowIndex + 1}, column ${colIndex + 1}, color ${value}`);
    cell.addEventListener("click", () => {
      guess.cells[rowIndex][colIndex] = guess.color;
      cell.className = `cell editor-cell cell-${guess.color}`;
      cell.title = `${PALETTE_NAMES[guess.color]} · ${guess.color}`;
      cell.setAttribute("aria-label", `Row ${rowIndex + 1}, column ${colIndex + 1}, color ${guess.color}`);
      state.submissions.delete(answerKey);
      status.textContent = "Draft · not checked";
      status.className = "answer-status";
    });
    editorGrid.append(cell);
  }));
  frame.append(editorGrid);
  builder.append(frame);

  const actions = document.createElement("div");
  actions.className = "builder-actions";
  const status = document.createElement("span");
  status.className = "answer-status";
  const submission = state.submissions.get(answerKey);
  if (submission?.result === "correct") {
    status.className = "answer-status is-correct";
    status.textContent = "Exact match · solved";
  } else if (submission?.result === "wrong") {
    status.className = "answer-status is-wrong";
    status.textContent = `Not correct yet · ${submission.attempts} attempt${submission.attempts === 1 ? "" : "s"}`;
  } else {
    status.textContent = "Draft · not checked";
  }
  const actionButtons = document.createElement("div");
  actionButtons.className = "builder-action-buttons";
  const clearButton = document.createElement("button");
  clearButton.type = "button";
  clearButton.className = "builder-clear";
  clearButton.textContent = "Clear";
  clearButton.addEventListener("click", () => {
    guess.cells = emptyGrid(guess.rows, guess.cols);
    state.submissions.delete(answerKey);
    renderTask();
  });
  const checkButton = document.createElement("button");
  checkButton.type = "button";
  checkButton.className = "check-button";
  checkButton.textContent = "Check guess";
  checkButton.addEventListener("click", () => {
    const attempts = (state.submissions.get(answerKey)?.attempts || 0) + 1;
    state.submissions.set(answerKey, { result: isExactMatch(guess, expectedGrid) ? "correct" : "wrong", attempts });
    renderTask();
  });
  actionButtons.append(clearButton, checkButton);
  actions.append(status, actionButtons);
  builder.append(actions);

  const reveal = document.createElement("button");
  reveal.type = "button";
  reveal.className = "reveal-button builder-reveal";
  reveal.textContent = "Reveal official answer";
  reveal.addEventListener("click", () => revealAnswer(answerKey));
  builder.append(reveal);
  figure.append(head, builder);
  return figure;
}

function revealAnswer(answerKey) {
  state.revealedAnswers.add(answerKey);
  renderTask();
}

function openFocusView(label, grid) {
  const rows = grid?.length || 0;
  const cols = grid?.[0]?.length || 0;
  els.focusTitle.textContent = label;
  els.focusMeta.textContent = `${rows} × ${cols} · task ${state.taskId}`;
  els.focusCanvas.replaceChildren(renderGridElement(grid, true));
  els.focusModal.hidden = false;
  document.body.classList.add("modal-open");
  requestAnimationFrame(fitFocusGrid);
  $("#close-focus").focus();
}

function fitFocusGrid() {
  if (els.focusModal.hidden) return;
  const grid = els.focusCanvas.querySelector(".focus-grid");
  if (!grid) return;
  const rows = Number(grid.style.getPropertyValue("--rows"));
  const cols = Number(grid.style.getPropertyValue("--cols"));
  const cell = Math.max(5, Math.floor(Math.min((els.focusCanvas.clientWidth - 42) / cols, (els.focusCanvas.clientHeight - 42) / rows)));
  grid.style.setProperty("--focus-cell", `${cell}px`);
}

function closeFocusView() {
  if (els.focusModal.hidden) return;
  els.focusModal.hidden = true;
  document.body.classList.remove("modal-open");
}

function renderArrow() {
  const arrow = document.createElement("div");
  arrow.className = "pair-arrow";
  arrow.setAttribute("aria-hidden", "true");
  arrow.textContent = "→";
  return arrow;
}

function moveTask(delta) {
  if (!state.filtered.length) return;
  state.index = (state.index + delta + state.filtered.length) % state.filtered.length;
  closeFocusView();
  renderQueue();
  loadCurrentTask();
}

function randomTask() {
  if (!state.filtered.length) return;
  let next = Math.floor(Math.random() * state.filtered.length);
  if (state.filtered.length > 1 && next === state.index) next = (next + 1) % state.filtered.length;
  state.index = next;
  closeFocusView();
  renderQueue();
  loadCurrentTask();
}

function filterTasks(value) {
  const query = value.trim().toLowerCase();
  state.filtered = state.files.filter((file) => file.toLowerCase().includes(query));
  state.index = 0;
  renderQueue();
  loadCurrentTask();
}

async function copyTaskId() {
  if (!state.taskId) return;
  try {
    await navigator.clipboard.writeText(state.taskId);
    const button = $("#copy-task-id");
    const original = button.innerHTML;
    button.innerHTML = "✓ Copied";
    setTimeout(() => { button.innerHTML = original; }, 1200);
  } catch (error) {
    console.warn("Clipboard unavailable", error);
  }
}

function bindEvents() {
  document.querySelectorAll(".dataset-tab").forEach((tab) => tab.addEventListener("click", () => {
    if (tab.dataset.dataset === state.dataset) return;
    closeFocusView();
    loadDataset(tab.dataset.dataset).catch(showError);
  }));
  els.search.addEventListener("input", (event) => filterTasks(event.target.value));
  $("#previous-task").addEventListener("click", () => moveTask(-1));
  $("#next-task").addEventListener("click", () => moveTask(1));
  $("#random-task").addEventListener("click", randomTask);
  $("#copy-task-id").addEventListener("click", copyTaskId);
  els.challengeToggle.addEventListener("click", () => {
    state.challenge = !state.challenge;
    localStorage.setItem("arc-explorer-challenge", String(state.challenge));
    state.revealedAnswers.clear();
    updateChallengeToggle();
    renderTask();
  });
  $("#close-focus").addEventListener("click", closeFocusView);
  $("#focus-backdrop").addEventListener("click", closeFocusView);
  window.addEventListener("resize", fitFocusGrid);
  document.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== els.search) {
      event.preventDefault();
      els.search.focus();
    }
    if (document.activeElement === els.search) return;
    if (event.key === "Escape") closeFocusView();
    if (event.key === "ArrowLeft") moveTask(-1);
    if (event.key === "ArrowRight") moveTask(1);
    if (event.key.toLowerCase() === "r") randomTask();
  });
}

function showError(error) {
  console.error(error);
  els.loading.hidden = true;
  els.explorer.hidden = true;
  els.error.hidden = false;
}

async function init() {
  bindEvents();
  try {
    await loadDataset("training");
    els.loading.hidden = true;
    els.explorer.hidden = false;
  } catch (error) {
    showError(error);
  }
}

init();
