/**
 * @module main.js
 *
 * This module provides functions to fetch and display game-related data, including:
 * - Fetching and displaying a list of games for a specific date.
 * - Populating player details in a specified container.
 * - Displaying play-by-play data for a game.
 * - Fetching and showing detailed information for a specific game.
 * - Live prediction display and auto-refresh for in-progress games.
 *
 * Functions:
 * - fetchAndUpdateGames(): Fetches games for a specific date and updates the games table.
 * - populatePlayerDetails(players, container, limit): Populates a container with player details.
 * - populatePlayByPlay(home_team, away_team, pbp, limit): Populates a table with play-by-play data.
 * - showGameDetails(gameId): Fetches and displays the details of a specific game.
 *
 * Each function handles specific aspects of game data presentation, aiming to provide a seamless
 * user experience by dynamically updating the UI with fetched data.
 */

// Track the auto-refresh interval so we can clear it if needed
let liveRefreshInterval = null;

/**
 * Fetches games for a specific date and updates the games table.
 * The date is read from the body's `data-query-date` attribute.
 * Each game is added as a new row in the games table.
 * @throws {Error} If the fetch operation fails.
 */
function fetchAndUpdateGames() {
  // Retrieve the query date from the body's dataset
  const queryDate = document.body.dataset.queryDate;

  // Get user's timezone from browser (IANA format, e.g., "America/New_York")
  const userTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

  // Fetch games data for the specified date, passing user timezone
  fetch(
    `/get-game-data?date=${queryDate}&user_tz=${encodeURIComponent(userTz)}`,
  )
    .then((response) => {
      if (!response.ok) {
        return response.json().then((error) => {
          throw new Error(error.error);
        });
      }
      return response.json();
    })
    .then((games) => {
      const tableBody = document.querySelector("#gamesTableBody");
      tableBody.innerHTML = ""; // Clear the table body

      let hasLiveGames = false;

      if (games.length > 0) {
        games.forEach((game) => {
          console.log("Game:", game);

          const isLive = game.game_status_code === 2;
          if (isLive) hasLiveGames = true;

          const row = document.createElement("tr");
          row.className = "game-row custom-vertical-align-middle";
          row.setAttribute("data-game-id", game.game_id);

          // Check if game is postponed
          const isPostponed = game.game_status === "PPD";
          const datetimeDisplay = isPostponed
            ? "PPD"
            : game.datetime_display.split("-").join("<br>");
          const homeScore = isPostponed ? "-" : game.home_score;
          const awayScore = isPostponed ? "-" : game.away_score;
          const openSpread = isPostponed ? "-" : game.opening_spread || "-";

          // For in-progress games with live predictions, show live values
          let predSpreadDisplay;
          let predWinnerDisplay;
          if (isPostponed) {
            predSpreadDisplay = "-";
            predWinnerDisplay = "-";
          } else if (isLive && game.live_spread) {
            // Show live spread with LIVE badge
            predSpreadDisplay = `<span class="live-badge">LIVE</span> ${game.live_spread}`;
            predWinnerDisplay = `<span class="live-badge">LIVE</span> ${game.live_winner || ""} ${game.live_win_pct || ""}`;
          } else {
            predSpreadDisplay = game.pred_spread || "-";
            predWinnerDisplay = `${game.pred_winner} ${game.pred_win_pct}`;
          }

          // Color-coding classes for completed games
          const winnerClass =
            game.pred_winner_correct === true
              ? " pred-correct"
              : game.pred_winner_correct === false
                ? " pred-wrong"
                : "";
          const spreadClass =
            game.spread_closer_than_vegas === true
              ? " pred-correct"
              : game.spread_closer_than_vegas === false
                ? " pred-wrong"
                : "";

          row.innerHTML = `
                        <td class="text-left custom-vertical-align-middle">${datetimeDisplay}</td>
                        <td class="custom-vertical-align-middle">
                            <div class="custom-display-flex custom-align-items-center">
                                <img src="${game.home_logo_url}" alt="Logo of ${
                                  game.home_team_display
                                }" class="custom-team-logo">
                                <div class="custom-text-align-left">${
                                  game.home_team_display
                                }</div>
                            </div>
                        </td>
                        <td class="custom-vertical-align-middle">
                            <div class="custom-display-flex custom-align-items-center">
                                <img src="${game.away_logo_url}" alt="Logo of ${
                                  game.away_team_display
                                }" class="custom-team-logo">
                                <div class="custom-text-align-left">${
                                  game.away_team_display
                                }</div>
                            </div>
                        </td>
                        <td class="text-center custom-vertical-align-middle">${openSpread}</td>
                        <td class="text-center custom-vertical-align-middle">${homeScore}</td>
                        <td class="text-center custom-vertical-align-middle">${awayScore}</td>
                        <td class="text-center custom-vertical-align-middle${spreadClass}">${predSpreadDisplay}</td>
                        <td class="text-center custom-vertical-align-middle${winnerClass}">${predWinnerDisplay}</td>
                    `;
          tableBody.appendChild(row);
        });
      } else {
        tableBody.innerHTML =
          '<tr><td colspan="8" class="text-center">No Games for the selected date</td></tr>';
      }

      // Auto-refresh: if there are live games, poll every 60 seconds
      if (hasLiveGames && !liveRefreshInterval) {
        liveRefreshInterval = setInterval(() => fetchAndUpdateGames(), 60000);
      } else if (!hasLiveGames && liveRefreshInterval) {
        // No more live games — stop polling
        clearInterval(liveRefreshInterval);
        liveRefreshInterval = null;
      }
    })
    .catch((error) => {
      console.error("Error fetching games:", error);
      const tableBody = document.querySelector("#gamesTableBody");
      tableBody.innerHTML = `<tr><td colspan="8" class="text-center">${error.message}</td></tr>`;
    });
}

/**
 * Populates a container with player details.
 *
 * For completed/in-progress games (status 2 or 3): shows actual points scored.
 * For upcoming games (status 1): shows predicted points if available, otherwise just the roster.
 *
 * @param {Array} players - An array of player objects with `player_headshot_url`, `player_name`, `points`, and `pred_points`.
 * @param {HTMLElement} container - The container to populate with player details.
 * @param {number} gameStatusCode - The game status code (1=scheduled, 2=in-progress, 3=final).
 * @param {number} [limit=5] - The maximum number of players to display. Defaults to 5.
 */
function populatePlayerDetails(players, container, gameStatusCode, limit = 5) {
  container.innerHTML = ""; // Clear previous content

  if (players.length === 0) {
    container.innerHTML =
      '<p class="text-muted text-center">No player data available</p>';
    return;
  }

  const hasActualStats = gameStatusCode === 2 || gameStatusCode === 3;
  const hasPredictions = players.some(
    (p) => p.pred_points !== null && p.pred_points !== undefined,
  );

  players.slice(0, limit).forEach((player) => {
    const playerDetailDiv = document.createElement("div");
    playerDetailDiv.className =
      "player-detail row d-flex align-items-center mb-3";

    let statsDisplay;
    if (
      hasActualStats &&
      player.points !== null &&
      player.points !== undefined
    ) {
      statsDisplay = `${player.points} PTS`;
    } else if (
      hasPredictions &&
      player.pred_points !== null &&
      player.pred_points !== undefined
    ) {
      statsDisplay = `${player.pred_points} PTS (pred)`;
    } else {
      statsDisplay = "";
    }

    playerDetailDiv.innerHTML = `
            <div class="col-auto">
                <img src="${player.player_headshot_url}" alt="${player.player_name}" class="img-fluid mb-2 player-headshot">
            </div>
            <div class="col">
                <p class="mb-0"><strong>${player.player_name}</strong></p>
                ${statsDisplay ? `<p class="mb-0">${statsDisplay}</p>` : ""}
            </div>
        `;
    container.appendChild(playerDetailDiv);
  });
}

/**
 * Populates a table with play-by-play data.
 *
 * @param {string} home_team - The name of the home team.
 * @param {string} away_team - The name of the away team.
 * @param {Array} pbp - An array of play-by-play records. Each record should have `time_info`, `description`, `home_score`, and `away_score` properties.
 * @param {number} [limit=Infinity] - The maximum number of records to display. Defaults to Infinity.
 */
function populatePlayByPlay(home_team, away_team, pbp, limit = Infinity) {
  const homeTeamHeader = document.getElementById("homeTeamHeader");
  const awayTeamHeader = document.getElementById("awayTeamHeader");
  const playByPlayBody = document.getElementById("playByPlayBody");

  homeTeamHeader.textContent = home_team;
  awayTeamHeader.textContent = away_team;

  playByPlayBody.innerHTML = ""; // Clear existing content

  if (pbp.length === 0) {
    playByPlayBody.innerHTML =
      '<tr><td colspan="4" class="text-center no-pbp-data">No Play By Play Logs Available</td></tr>';
  } else {
    pbp.slice(0, limit).forEach((record) => {
      const row = document.createElement("tr");
      row.innerHTML = `
                <td>${record.time_info}</td>
                <td>${record.description}</td>
                <td>${record.home_score}</td>
                <td>${record.away_score}</td>
            `;
      playByPlayBody.appendChild(row);
    });
  }
}

/**
 * Fetches and displays the details of a specific game.
 *
 * @param {string} gameId - The ID of the game to fetch details for.
 */
function showGameDetails(gameId) {
  // Get user's timezone from browser (IANA format, e.g., "America/New_York")
  const userTz = Intl.DateTimeFormat().resolvedOptions().timeZone;

  fetch(
    `/get-game-data?game_id=${gameId}&user_tz=${encodeURIComponent(userTz)}`,
  )
    .then((response) => {
      if (!response.ok) {
        return response.json().then((error) => {
          throw new Error(error.error);
        });
      }
      return response.json();
    })
    .then((data) => {
      const game = data[0];

      const {
        home,
        away,
        home_full_name: homeFullName,
        away_full_name: awayFullName,
        home_logo_url: homeLogoUrl,
        away_logo_url: awayLogoUrl,
        home_score: homeScore,
        away_score: awayScore,
        game_status_code: gameStatusCode,
        datetime_display: dateTimeDisplay,
        condensed_pbp: playByPlay,
        home_players: homePlayers,
        away_players: awayPlayers,
        pred_winner: predictedWinner,
        pred_win_pct: predictedWinPercentage,
      } = game;

      const isLive = gameStatusCode === 2;

      // Modal title — show score for completed games, time for upcoming
      const modalTitle = document.querySelector("#gameDetailsModalLabel");
      const scoreDisplay =
        gameStatusCode === 3
          ? `${homeScore} - ${awayScore}`
          : gameStatusCode === 2
            ? `${homeScore} - ${awayScore} <span class="live-badge">LIVE</span>`
            : "vs";
      modalTitle.innerHTML = `
                ${homeFullName} <img src="${homeLogoUrl}" alt="${homeFullName}" class="team-logo">
                ${scoreDisplay}
                <img src="${awayLogoUrl}" alt="${awayFullName}" class="team-logo"> ${awayFullName}
                <span class="breakpoint"> - <wbr></span>${dateTimeDisplay}
            `;

      const template = document
        .querySelector("#gameDetailsTemplate")
        .content.cloneNode(true);
      template.querySelector("#templateHomeTeam").textContent = home;
      template.querySelector("#templateAwayTeam").textContent = away;
      template.querySelector("#templateHomeLogo").src = homeLogoUrl;
      template.querySelector("#templateAwayLogo").src = awayLogoUrl;

      // Spreads
      template.querySelector("#templateOpenSpread").textContent =
        game.opening_spread || "-";
      template.querySelector("#templatePredictedSpread").textContent =
        game.pred_spread || "-";

      // Show actual margin for completed games
      if (gameStatusCode === 3 && homeScore !== "" && awayScore !== "") {
        const actualMargin = homeScore - awayScore;
        const marginStr = (actualMargin >= 0 ? "+" : "") + actualMargin;
        template.querySelector("#templateActualMargin").textContent = marginStr;
        template.querySelector("#templateResultSection").style.display =
          "block";
      }

      // For live games, show live spread and progress in the modal
      if (isLive && game.live_spread) {
        const liveSection = template.querySelector("#templateLiveSection");
        if (liveSection) {
          liveSection.style.display = "block";
          template.querySelector("#templateLiveSpread").textContent =
            game.live_spread;
          template.querySelector("#templateLiveWinPct").textContent =
            `${game.live_winner || ""} ${game.live_win_pct || ""}`;
          template.querySelector("#templateGameProgress").textContent =
            game.game_progress || "";
        }
      }

      // Winner
      template.querySelector("#templatePredictedWinPct").textContent =
        predictedWinPercentage || "";

      populatePlayerDetails(
        homePlayers,
        template.querySelector("#homeTeamPlayers"),
        gameStatusCode,
      );
      populatePlayerDetails(
        awayPlayers,
        template.querySelector("#awayTeamPlayers"),
        gameStatusCode,
      );

      const modalBody = document.querySelector("#gameDetailsModal .modal-body");
      modalBody.innerHTML = "";
      modalBody.appendChild(template);

      const winnerLeftIcon = document.getElementById("winnerLeftIcon");
      const winnerRightIcon = document.getElementById("winnerRightIcon");

      if (predictedWinner === home) {
        winnerLeftIcon.style.visibility = "visible";
        winnerRightIcon.style.visibility = "hidden";
      } else if (predictedWinner === away) {
        winnerRightIcon.style.visibility = "visible";
        winnerLeftIcon.style.visibility = "hidden";
      } else {
        winnerLeftIcon.style.visibility = "hidden";
        winnerRightIcon.style.visibility = "hidden";
      }

      populatePlayByPlay(home, away, playByPlay, 100);

      var gameDetailsModal = new bootstrap.Modal(
        document.getElementById("gameDetailsModal"),
        {},
      );
      gameDetailsModal.show();
    })
    .catch((error) => {
      console.error("Error fetching game details:", error);
    });
}

// Initialize event listeners after DOM content is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  fetchAndUpdateGames();

  document
    .querySelector("#gamesTableBody")
    .addEventListener("click", function (event) {
      let target = event.target;

      while (target && !target.classList.contains("game-row")) {
        target = target.parentElement;
      }

      if (target && target.classList.contains("game-row")) {
        const gameId = target.getAttribute("data-game-id");
        showGameDetails(gameId);
      }
    });
});
