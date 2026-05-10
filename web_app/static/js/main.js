/**
 * Music Recommender - Frontend JavaScript
 * Handles search, filtering, and YouTube player integration
 */

// =============================================================================
// State Management
// =============================================================================

const state = {
    currentMode: 'search', // 'search', 'image', 'mood', 'context', 'history', 'playlists'
    isLoading: false,
    results: [],
    selectedImage: null,
    recentSearches: JSON.parse(localStorage.getItem('recentSearches')) || [],
    // Auth
    user: null,
    token: localStorage.getItem('authToken') || null,
    // Playlists
    playlists: [],
    currentPickerSong: null, // song being added to playlist
    // Search-First Flow
    allSearchResults: [],   // full list from /api/song-search (client-side pagination)
    searchPage: 1,          // current page (1-indexed)
    totalSearchPages: 0,    // computed from allSearchResults.length
    lastSearchQuery: '',    // remembered to restore on back
    lastSearchArtist: '',   // remembered to restore on back
    selectedSong: null,     // { song, artist } of the song whose recommendations are shown
};

// Number of search result cards shown per page
const SEARCH_PAGE_SIZE = 12;

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
    // Inputs
    searchInput: () => document.getElementById('searchInput'),
    artistInput: () => document.getElementById('artistInput'),
    searchBtn: () => document.getElementById('searchBtn'),
    autocompleteDropdown: () => document.getElementById('autocompleteDropdown'),

    // Search-First: search results section
    searchResultsSection: () => document.getElementById('searchResultsSection'),
    searchResultsList: () => document.getElementById('searchResultsList'),
    searchResultsTitle: () => document.getElementById('searchResultsTitle'),
    searchResultsCount: () => document.getElementById('searchResultsCount'),
    searchResultsStats: () => document.getElementById('searchResultsStats'),
    searchResultsBreadcrumb: () => document.getElementById('searchResultsBreadcrumb'),
    breadcrumbCurrent: () => document.getElementById('breadcrumbCurrent'),
    paginationPrev: () => document.getElementById('paginationPrev'),
    paginationNext: () => document.getElementById('paginationNext'),
    paginationPages: () => document.getElementById('paginationPages'),

    // Results (recommendations)
    loading: () => document.getElementById('loading'),
    resultsSection: () => document.getElementById('resultsSection'),
    resultsGrid: () => document.getElementById('resultsGrid'),
    resultsTitle: () => document.getElementById('resultsTitle'),
    resultsCount: () => document.getElementById('resultsCount'),

    // Modal
    youtubeModal: () => document.getElementById('youtubeModal'),
    youtubePlayer: () => document.getElementById('youtubePlayer'),
    modalSongTitle: () => document.getElementById('modalSongTitle'),
    modalArtistName: () => document.getElementById('modalArtistName'),

    // Navigation
    navLinks: () => document.querySelectorAll('.nav-link'),

    // Image Upload
    imageInput: () => document.getElementById('imageInput'),
    imageDropZone: () => document.getElementById('imageDropZone'),
    dropZoneContent: () => document.getElementById('dropZoneContent'),
    imagePreview: () => document.getElementById('imagePreview'),
    imageActions: () => document.getElementById('imageActions'),
    imageMode: () => document.getElementById('image-mode'),
    detectedMoodContainer: () => document.getElementById('detectedMoodContainer'),
    detectedIcon: () => document.getElementById('detectedIcon'),
    detectedLabel: () => document.getElementById('detectedLabel'),
    detectedType: () => document.getElementById('detectedType'),
    confidenceFill: () => document.getElementById('confidenceFill'),
    confidenceText: () => document.getElementById('confidenceText')
};

// =============================================================================
// Mode Switching
// =============================================================================

function switchMode(mode) {
    state.currentMode = mode;

    // Update nav links
    elements.navLinks().forEach(link => {
        link.classList.toggle('active', link.dataset.mode === mode);
    });

    // Show/hide appropriate sections
    const modes = ['search', 'image', 'mood', 'context', 'history', 'playlists'];
    modes.forEach(m => {
        const section = document.getElementById(`${m}-mode`);
        if (section) {
            section.classList.toggle('hidden', m !== mode);
        }
    });

    // Clear results when switching modes
    hideResults();

    // Clear image preview when switching modes
    if (mode !== 'image') {
        clearImage();
    }

    // Hide detected mood container when switching modes
    if (elements.detectedMoodContainer()) {
        elements.detectedMoodContainer().classList.add('hidden');
    }

    // Load data for protected tabs
    if (mode === 'history') loadHistory();
    if (mode === 'playlists') loadPlaylists();
}

// =============================================================================
// Search, Autocomplete & History
// =============================================================================

let autocompleteTimeout;

function updateRecentSearches(query, artist = '') {
    if (!query) return;
    
    const entry = { song: query, artist: artist };
    
    // Remove if already exists
    state.recentSearches = state.recentSearches.filter(
        item => !(item.song.toLowerCase() === query.toLowerCase() && (item.artist || '').toLowerCase() === artist.toLowerCase())
    );
    
    // Add to beginning
    state.recentSearches.unshift(entry);
    
    if (state.recentSearches.length > 10) {
        state.recentSearches.pop();
    }
    
    localStorage.setItem('recentSearches', JSON.stringify(state.recentSearches));
}

function showRecentSearches() {
    const dropdown = elements.autocompleteDropdown();
    
    if (state.recentSearches.length === 0) {
        dropdown.classList.add('hidden');
        return;
    }
    
    let html = '<div class="autocomplete-header">Recent Searches</div>';
    html += state.recentSearches.map(item => `
        <div class="autocomplete-item" onclick="selectAutocomplete('${escapeHtml(item.song)}', '${escapeHtml(item.artist || '')}')">
            <span class="autocomplete-icon">🕒</span>
            <div class="autocomplete-text">
                <div class="autocomplete-song">${escapeHtml(item.song)}</div>
                ${item.artist ? `<div class="autocomplete-artist">${escapeHtml(item.artist)}</div>` : ''}
            </div>
        </div>
    `).join('');
    
    dropdown.innerHTML = html;
    dropdown.classList.remove('hidden');
}

async function handleAutocompleteInput() {
    const query = elements.searchInput().value.trim();
    const dropdown = elements.autocompleteDropdown();
    
    clearTimeout(autocompleteTimeout);
    
    if (!query) {
        showRecentSearches();
        return;
    }
    
    autocompleteTimeout = setTimeout(async () => {
        try {
            // Get selected model
            const searchModelInput = document.querySelector('input[name="searchModel"]:checked');
            const selectedModel = searchModelInput ? searchModelInput.value : 'model1';
            
            const url = new URL('/api/autocomplete', window.location.origin);
            url.searchParams.set('q', query);
            url.searchParams.set('model', selectedModel);
            
            const response = await fetch(url);
            if (!response.ok) throw new Error('Failed to fetch suggestions');
            
            const data = await response.json();
            
            if (data.length === 0) {
                dropdown.classList.add('hidden');
                return;
            }
            
            let html = '<div class="autocomplete-header">Suggestions</div>';
            html += data.map(item => `
                <div class="autocomplete-item" onclick="selectAutocomplete('${escapeHtml(item.song)}', '${escapeHtml(item.artist || '')}')">
                    <span class="autocomplete-icon">🔍</span>
                    <div class="autocomplete-text">
                        <div class="autocomplete-song">${escapeHtml(item.song)}</div>
                        ${item.artist ? `<div class="autocomplete-artist">${escapeHtml(item.artist)}</div>` : ''}
                    </div>
                </div>
            `).join('');
            
            dropdown.innerHTML = html;
            dropdown.classList.remove('hidden');
            
        } catch (error) {
            console.error('Autocomplete error:', error);
            dropdown.classList.add('hidden');
        }
    }, 300);
}

function selectAutocomplete(song, artist) {
    elements.searchInput().value = song;
    elements.artistInput().value = artist || '';
    elements.autocompleteDropdown().classList.add('hidden');
    handleSearch(); // goes through song-search → results list
}

/**
 * Step 1 of the Search-First flow.
 * Calls /api/song-search → shows matching songs list with pagination.
 * User then clicks a card to trigger findSimilarSongs() (Step 2).
 */
async function handleSearch() {
    const query = elements.searchInput().value.trim();
    const artist = elements.artistInput().value.trim();

    // Get selected model
    const searchModelInput = document.querySelector('input[name="searchModel"]:checked');
    const selectedModel = searchModelInput ? searchModelInput.value : 'model1';

    if (!query) {
        showToast('Please enter a song name', 'warning');
        elements.searchInput().focus();
        return;
    }

    showLoading();
    // Hide any previous recommendation results
    hideResults();
    // Hide search results section while loading
    if (elements.searchResultsSection()) {
        elements.searchResultsSection().classList.add('hidden');
    }

    try {
        const url = new URL('/api/song-search', window.location.origin);
        url.searchParams.set('q', query);
        if (artist) url.searchParams.set('artist', artist);
        url.searchParams.set('model', selectedModel);

        const response = await fetch(url.toString());
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Song search failed');
        }

        // Save to state for pagination + back navigation
        state.allSearchResults = data.results || [];
        state.searchPage = 1;
        state.totalSearchPages = Math.ceil(state.allSearchResults.length / SEARCH_PAGE_SIZE);
        state.lastSearchQuery = query;
        state.lastSearchArtist = artist;
        state.selectedSong = null;

        if (state.allSearchResults.length === 0) {
            showToast(`No songs found for "${query}"`, 'warning');
            hideLoading();
            return;
        }

        // Render the search results section
        const modelName = selectedModel === 'model2' ? 'Model 2 (Audio-Only)' : 'Model 1 (Hybrid)';
        displaySearchResults(data, modelName);
        updateRecentSearches(query, artist);
        elements.autocompleteDropdown().classList.add('hidden');

    } catch (error) {
        console.error('Song search error:', error);
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

// =============================================================================
// Search-First Flow — Step 1: Display matching songs
// =============================================================================

/**
 * Renders the search results section header + first page.
 * @param {object} data  - API response from /api/song-search
 * @param {string} modelName - human-readable model label
 */
function displaySearchResults(data, modelName) {
    const section = elements.searchResultsSection();
    if (!section) return;

    // Update header
    const query = data.query || state.lastSearchQuery;
    elements.searchResultsTitle().textContent = `Results for "${query}"`;
    elements.searchResultsCount().textContent =
        `${data.total} song${data.total !== 1 ? 's' : ''} · ${modelName}`;

    // Stats: originals · remixes
    const statsEl = elements.searchResultsStats();
    if (statsEl) {
        statsEl.innerHTML = [
            data.originals_count > 0
                ? `<span class="stats-original-dot"></span>${data.originals_count} original${data.originals_count !== 1 ? 's' : ''}`
                : null,
            data.remixes_count > 0
                ? `<span class="stats-remix-dot"></span>${data.remixes_count} remix${data.remixes_count !== 1 ? 'es' : ''}`
                : null,
        ].filter(Boolean).join(' &middot; ');
    }

    // Hide breadcrumb (only visible when showing recommendations)
    const breadcrumb = elements.searchResultsBreadcrumb();
    if (breadcrumb) breadcrumb.classList.add('hidden');

    // Show hint
    const hint = section.querySelector('.search-results-hint');
    if (hint) hint.classList.remove('hidden');

    section.classList.remove('hidden');
    displaySearchPage(1);
}

/**
 * Renders page `page` of state.allSearchResults into the list + updates pagination.
 * @param {number} page - 1-indexed page number
 */
function displaySearchPage(page) {
    if (page < 1 || (state.totalSearchPages > 0 && page > state.totalSearchPages)) return;

    state.searchPage = page;
    const start = (page - 1) * SEARCH_PAGE_SIZE;
    const pageItems = state.allSearchResults.slice(start, start + SEARCH_PAGE_SIZE);

    const list = elements.searchResultsList();
    if (!list) return;

    list.innerHTML = pageItems
        .map((song, i) => createSearchResultCard(song, start + i))
        .join('');

    // Staggered entrance animation
    list.querySelectorAll('.search-result-card').forEach((card, i) => {
        card.style.animationDelay = `${i * 30}ms`;
        card.classList.add('entering');
    });

    renderPagination(page, state.totalSearchPages);

    // Scroll to top of the search results section
    const section = elements.searchResultsSection();
    if (section) {
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Navigate to a given page (called by pagination buttons).
 * @param {number} page
 */
function goToSearchPage(page) {
    if (page < 1 || page > state.totalSearchPages) return;
    displaySearchPage(page);
}

/**
 * Renders the pagination bar.
 * Shows up to 5 page numbers in a sliding window with ellipsis.
 * @param {number} current - current page (1-indexed)
 * @param {number} total   - total pages
 */
function renderPagination(current, total) {
    const prev = elements.paginationPrev();
    const next = elements.paginationNext();
    const pages = elements.paginationPages();
    const nav = document.getElementById('searchPagination');

    if (!prev || !next || !pages) return;

    // Hide pagination if only 1 page
    if (total <= 1) {
        if (nav) nav.classList.add('hidden');
        return;
    }
    if (nav) nav.classList.remove('hidden');

    prev.disabled = current <= 1;
    next.disabled = current >= total;

    // Build page number buttons (sliding window of 5)
    const WINDOW = 5;
    let startPage = Math.max(1, current - Math.floor(WINDOW / 2));
    let endPage   = Math.min(total, startPage + WINDOW - 1);
    // Shift window if at the end
    if (endPage - startPage < WINDOW - 1) {
        startPage = Math.max(1, endPage - WINDOW + 1);
    }

    let html = '';
    if (startPage > 1) {
        html += `<button class="pagination-page" onclick="goToSearchPage(1)">1</button>`;
        if (startPage > 2) html += `<span class="pagination-ellipsis">…</span>`;
    }
    for (let p = startPage; p <= endPage; p++) {
        html += `<button class="pagination-page${p === current ? ' active' : ''}" 
                    onclick="goToSearchPage(${p})"
                    aria-label="Page ${p}"
                    aria-current="${p === current ? 'page' : 'false'}">${p}</button>`;
    }
    if (endPage < total) {
        if (endPage < total - 1) html += `<span class="pagination-ellipsis">…</span>`;
        html += `<button class="pagination-page" onclick="goToSearchPage(${total})">${total}</button>`;
    }
    pages.innerHTML = html;
}

/**
 * Creates a compact search-result card HTML string.
 * Click → findSimilarSongs(); Play button → playSong().
 */
function createSearchResultCard(song, globalIndex) {
    const songE   = escapeHtml(song.song);
    const artistE = escapeHtml(song.artist);
    const genreE  = escapeHtml(song.genre || '');
    const rankNum = globalIndex + 1;

    const badge = song.is_remix
        ? '<span class="tag-remix">Remix</span>'
        : '<span class="tag-original">Original</span>';

    const genreTag = genreE
        ? `<span class="src-genre-tag">${genreE}</span>`
        : '';

    return `
        <div class="search-result-card"
             role="button"
             tabindex="0"
             aria-label="${songE} by ${artistE} — click to find similar songs"
             onclick="findSimilarSongs('${songE}', '${artistE}')"
             onkeydown="if(event.key==='Enter'||event.key===' ')findSimilarSongs('${songE}','${artistE}')">
            <span class="src-rank">${rankNum}</span>
            <div class="src-info">
                <div class="src-song-name" title="${songE}">${songE}</div>
                <div class="src-artist">${artistE}</div>
            </div>
            <div class="src-tags">
                ${badge}
                ${genreTag}
            </div>
            <button class="src-play-btn"
                    title="Play on YouTube"
                    aria-label="Play ${songE} on YouTube"
                    onclick="event.stopPropagation(); playSong('${songE}', '${artistE}')">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M8 5v14l11-7z"/>
                </svg>
            </button>
            <svg class="src-arrow" width="16" height="16" viewBox="0 0 24 24"
                 fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
        </div>`;
}

// =============================================================================
// Search-First Flow — Step 2: Recommendations for the selected song
// =============================================================================

/**
 * Step 2: fetch /api/search for similar songs to the clicked song,
 * display them in the main results section, and show the Back button.
 */
async function findSimilarSongs(song, artist) {
    state.selectedSong = { song, artist };

    const searchModelInput = document.querySelector('input[name="searchModel"]:checked');
    const selectedModel = searchModelInput ? searchModelInput.value : 'model1';

    showLoading();

    try {
        const url = new URL('/api/search', window.location.origin);
        url.searchParams.set('q', song);
        if (artist) url.searchParams.set('artist', artist);
        url.searchParams.set('model', selectedModel);

        const response = await apiFetch(url.toString());
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        const modelName = selectedModel === 'model2' ? 'Model 2 (Audio-Only)' : 'Model 1 (Hybrid)';
        showRecommendationsView(song, artist, data.results, data.count, modelName);

    } catch (error) {
        console.error('findSimilarSongs error:', error);
        showToast(error.message, 'error');
    } finally {
        hideLoading();
    }
}

/**
 * Switch the Search Results section into "recommendations" view:
 * - Hide the song list + pagination + hint
 * - Show breadcrumb with Back button
 * - Show recommendation results below
 */
function showRecommendationsView(song, artist, results, count, modelName) {
    const section = elements.searchResultsSection();
    if (!section) return;

    // Show breadcrumb
    const breadcrumb = elements.searchResultsBreadcrumb();
    if (breadcrumb) breadcrumb.classList.remove('hidden');
    const crumb = elements.breadcrumbCurrent();
    if (crumb) crumb.textContent = `"${song}" — ${artist}`;

    // Update header to say "Recommendations"
    elements.searchResultsTitle().textContent = `Similar to "${song}"`;
    elements.searchResultsCount().textContent = `${count} recommendation${count !== 1 ? 's' : ''} · ${modelName}`;

    // Clear stats and hide hint
    const stats = elements.searchResultsStats();
    if (stats) stats.innerHTML = '';
    const hint = section.querySelector('.search-results-hint');
    if (hint) hint.classList.add('hidden');

    // Hide the compact list + pagination
    const list = elements.searchResultsList();
    if (list) list.innerHTML = '';
    const nav = document.getElementById('searchPagination');
    if (nav) nav.classList.add('hidden');

    // Show recommendations in the main results section
    displayResults(results, `Similar to "${song}" [${modelName}]`, count);

    // Scroll to results
    elements.resultsSection().scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Back button handler: return to the search results list at the remembered page.
 */
function showSearchResultsView() {
    // Hide recommendations
    hideResults();

    const section = elements.searchResultsSection();
    if (!section) return;

    // Hide breadcrumb
    const breadcrumb = elements.searchResultsBreadcrumb();
    if (breadcrumb) breadcrumb.classList.add('hidden');

    // Restore header
    const query = state.lastSearchQuery;
    elements.searchResultsTitle().textContent = `Results for "${query}"`;
    elements.searchResultsCount().textContent =
        `${state.allSearchResults.length} song${state.allSearchResults.length !== 1 ? 's' : ''}`;

    // Show hint
    const hint = section.querySelector('.search-results-hint');
    if (hint) hint.classList.remove('hidden');

    // Re-render the last page the user was on
    displaySearchPage(state.searchPage);
    section.classList.remove('hidden');

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function searchByMood(mood) {
    showLoading();

    try {
        const response = await fetch(`/api/mood/${mood}?limit=10`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        displayResults(data, `${mood} Mood`, data.length);

    } catch (error) {
        console.error('Mood search error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        hideLoading();
    }
}

async function searchByContext(context) {
    showLoading();

    try {
        const response = await fetch(`/api/context/${context}?limit=10`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to get recommendations');
        }

        displayResults(data, `${context} Vibes`, data.length);

    } catch (error) {
        console.error('Context search error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        hideLoading();
    }
}


// =============================================================================
// Image Upload Functions
// =============================================================================

function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        previewImage(file);
    }
}

function previewImage(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    if (!allowedTypes.includes(file.type)) {
        showToast('Please upload a JPEG, PNG, WebP, or GIF image', 'error');
        return;
    }

    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showToast('Image too large. Maximum size is 10MB', 'error');
        return;
    }

    // Store the file
    state.selectedImage = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = elements.imagePreview();
        const dropZoneContent = elements.dropZoneContent();
        const actions = elements.imageActions();

        preview.src = e.target.result;
        preview.classList.remove('hidden');
        dropZoneContent.classList.add('hidden');
        actions.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

function clearImage() {
    state.selectedImage = null;

    const preview = elements.imagePreview();
    const dropZoneContent = elements.dropZoneContent();
    const actions = elements.imageActions();
    const input = elements.imageInput();
    const detectedContainer = elements.detectedMoodContainer();

    if (preview) {
        preview.src = '';
        preview.classList.add('hidden');
    }
    if (dropZoneContent) dropZoneContent.classList.remove('hidden');
    if (actions) actions.classList.add('hidden');
    if (input) input.value = '';
    if (detectedContainer) detectedContainer.classList.add('hidden');
}

async function handleImageUpload() {
    if (!state.selectedImage) {
        showToast('Please select an image first', 'warning');
        return;
    }

    // Bug 3 fix: Show specific loading message for CLIP analysis
    showLoading('🤖 Analyzing image with AI... (may take up to 30s on first use)');

    // Bug 3 fix: Warn user if taking long (CLIP lazy-loading ~350MB model)
    const slowTimer = setTimeout(() => {
        showToast('⏳ AI model is loading for the first time (~350MB). Please wait...', 'warning');
    }, 8000);

    // Hide detected mood container from previous analysis
    if (elements.detectedMoodContainer()) {
        elements.detectedMoodContainer().classList.add('hidden');
    }

    try {
        // Create FormData and append image
        const formData = new FormData();
        formData.append('file', state.selectedImage);

        // Get selected model
        const selectedModel = document.querySelector('input[name="imageModel"]:checked').value;
        const endpoint = selectedModel === 'model2' ? '/api/recommend/image-v2' : '/api/recommend-from-image';

        // Send to backend
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Failed to analyze image');
        }

        if (!data.success) {
            showToast(data.message || 'Failed to analyze image', 'error');
            hideResults();
            return;
        }

        // Display detected mood/context
        displayDetectedMood(data);

        // Display recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            const modelName = selectedModel === 'model2' ? 'Model 2 (Audio)' : 'Model 1 (Hybrid)';
            displayResults(
                data.recommendations,
                `${data.detected_label} ${data.detected_type === 'mood' ? 'Mood' : 'Activity'} Songs [${modelName}]`,
                data.recommendations.length
            );
        } else {
            showToast('No recommendations found for this mood/context', 'warning');
            hideResults();
        }

    } catch (error) {
        console.error('Image upload error:', error);
        showToast(error.message, 'error');
        hideResults();
    } finally {
        clearTimeout(slowTimer);  // Bug 3 fix: cancel slow-load warning
        hideLoading();
    }
}

function displayDetectedMood(data) {
    const container = elements.detectedMoodContainer();
    const icon = elements.detectedIcon();
    const label = elements.detectedLabel();
    const type = elements.detectedType();
    const confidenceFill = elements.confidenceFill();
    const confidenceText = elements.confidenceText();

    if (!container) return;

    // Set emoji based on detected label
    const moodEmojis = {
        'Happy': '😊',
        'Sad': '😢',
        'Energetic': '⚡',
        'Calm': '😌',
        'Angry': '😠',
        'Party': '🎉',
        'Workout': '💪',
        'Study': '📚',
        'Relax': '🌸',
        'Driving': '🚗'
    };

    icon.textContent = moodEmojis[data.detected_label] || '🎵';
    label.textContent = data.detected_label;
    type.textContent = data.detected_type === 'mood' ? 'Mood detected' : 'Activity detected';

    const confidencePercent = Math.round(data.confidence * 100);
    confidenceFill.style.width = `${confidencePercent}%`;
    confidenceText.textContent = `${confidencePercent}% confidence`;

    container.classList.remove('hidden');
}

function setupImageUploadListeners() {
    const imageInput = elements.imageInput();
    const dropZone = elements.imageDropZone();

    if (!imageInput || !dropZone) return;

    // File input change
    imageInput.addEventListener('change', handleImageSelect);

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');

        const file = e.dataTransfer.files[0];
        if (file) {
            previewImage(file);
        }
    });
}

function displayResults(results, title, count) {
    state.results = results;

    if (!results || results.length === 0) {
        elements.resultsGrid().innerHTML = `
            <div class="no-results" style="grid-column: 1/-1; text-align: center; padding: 60px 20px; color: var(--text-secondary);">
                <p style="font-size: 48px; margin-bottom: 16px;">?</p>
                <p style="font-size: 18px;">No songs found</p>
                <p style="font-size: 14px; margin-top: 8px;">Try a different search term or filter</p>
            </div>
        `;
        elements.resultsTitle().textContent = title;
        elements.resultsCount().textContent = '0 songs';
        elements.resultsSection().classList.remove('hidden');
        return;
    }

    elements.resultsTitle().textContent = title;
    elements.resultsCount().textContent = `${count} songs`;

    elements.resultsGrid().innerHTML = results.map((song, index) => createSongCard(song, index)).join('');

    elements.resultsSection().classList.remove('hidden');

    // Animate cards
    const cards = elements.resultsGrid().querySelectorAll('.song-card');
    cards.forEach((card, i) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        setTimeout(() => {
            card.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, i * 50);
    });

    // Enrich cards with Spotify album art (non-blocking, runs in background)
    enrichSongCards(results);
}

/**
 * Fetch Spotify metadata for each song in results and inject album art + links.
 * Runs asynchronously so cards appear immediately, art loads progressively.
 */
async function enrichSongCards(results) {
    const enrichPromises = results.map(async (song, index) => {
        try {
            const url = new URL('/api/spotify-enrich', window.location.origin);
            url.searchParams.set('q', song.song);
            if (song.artist) url.searchParams.set('artist', song.artist);

            const res = await fetch(url);
            if (!res.ok) return;
            const data = await res.json();

            // Find the card by data-index
            const card = elements.resultsGrid().querySelector(`[data-song-index="${index}"]`);
            if (!card) return;

            // Bug 2 fix: Always show Spotify button — use exact URL if found, search URL as fallback
            const spotifyBtn = card.querySelector('.spotify-link-btn');
            if (spotifyBtn) {
                const finalSpotifyUrl = data.spotify_url || data.spotify_search_url;
                if (finalSpotifyUrl) {
                    spotifyBtn.href = finalSpotifyUrl;
                    spotifyBtn.style.display = 'flex';
                    // Add visual hint that this is a search (not exact) link
                    if (!data.spotify_url && data.spotify_search_url) {
                        spotifyBtn.title = 'Search on Spotify (no exact match found)';
                        spotifyBtn.style.opacity = '0.7';
                    }
                }
            }

            if (!data.found) return; // Skip album art / metadata injection if not found

            // Inject album art
            if (data.album_art) {
                const placeholder = card.querySelector('.album-art-placeholder');
                if (placeholder) {
                    placeholder.innerHTML = `<img src="${data.album_art}" alt="${escapeHtml(song.song)} album art" class="album-art-img" loading="lazy">`;
                    placeholder.classList.add('loaded');
                }
            }

            // Add explicit badge if needed
            if (data.explicit) {
                const tags = card.querySelector('.song-tags');
                if (tags && !tags.querySelector('.tag-explicit')) {
                    tags.insertAdjacentHTML('beforeend', '<span class="tag tag-explicit">E</span>');
                }
            }

            // Inject duration and popularity
            const infoDiv = card.querySelector('.song-info');
            if (infoDiv) {
                if (data.duration_ms) {
                    const durSpan = infoDiv.querySelector('.song-duration');
                    if (durSpan) durSpan.textContent = ` • ${formatDuration(data.duration_ms)}`;
                }
                if (data.popularity) {
                    const popSpan = infoDiv.querySelector('.popularity-badge');
                    if (popSpan) {
                        popSpan.innerHTML = `🔥 ${data.popularity}`;
                        popSpan.style.fontSize = '12px';
                        popSpan.style.color = 'var(--text-muted)';
                        popSpan.classList.remove('hidden');
                    }
                }
            }

            // Inject 30s preview audio
            if (data.preview_url) {
                const previewContainer = card.querySelector('.preview-container');
                if (previewContainer) {
                    previewContainer.innerHTML = `
                        <div class="audio-preview" onclick="event.stopPropagation()">
                            <audio controls preload="none">
                                <source src="${data.preview_url}" type="audio/mpeg">
                            </audio>
                        </div>
                    `;
                    previewContainer.classList.remove('hidden');
                }
            }
            // Store enriched data on card dataset for Add-to-Playlist
            if (data.spotify_url) card.dataset.spotifyUrl = data.spotify_url;
            if (data.album_art) card.dataset.albumArt = data.album_art;

        } catch (e) {
            // Silently fail — enrichment is optional
        }
    });

    // Run all enrichments in parallel (but don't wait)
    Promise.allSettled(enrichPromises);
}

function formatDuration(ms) {
    if (!ms) return '';
    const minutes = Math.floor(ms / 60000);
    const seconds = ((ms % 60000) / 1000).toFixed(0);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
}

function createSongCard(song, index) {
    const similarity = song.similarity !== null && song.similarity !== undefined
        ? `<div class="similarity-score">
               Similarity: ${(song.similarity * 100).toFixed(1)}%
               <div class="similarity-bar">
                   <div class="similarity-fill" style="width: ${song.similarity * 100}%"></div>
               </div>
           </div>`
        : '';

    // Escape values for inline onclick attributes
    const songE = escapeHtml(song.song);
    const artistE = escapeHtml(song.artist);
    const genreE = escapeHtml(song.genre);
    const emotionE = escapeHtml(song.emotion);

    return `
        <div class="song-card" data-song-index="${index}"
             data-song="${songE}" data-artist="${artistE}"
             data-genre="${genreE}" data-emotion="${emotionE}"
             data-spotify-url="" data-album-art=""
             onclick="playSong('${songE}', '${artistE}')">
            <div class="album-art-placeholder">
                <span class="album-art-note">🎵</span>
            </div>
            <div class="song-body">
                <div class="song-header">
                    <div class="song-info">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <h3 class="song-name" title="${songE}">${songE}</h3>
                            <span class="popularity-badge hidden"></span>
                        </div>
                        <p class="song-artist" title="${artistE}">
                            ${artistE}<span class="song-duration"></span>
                        </p>
                    </div>
                    <div class="song-actions">
                        <a class="spotify-link-btn" href="#" target="_blank" rel="noopener noreferrer"
                            title="Open on Spotify" style="display:none"
                            onclick="event.stopPropagation()">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
                            </svg>
                        </a>
                        <button class="add-to-playlist-btn" title="Add to Playlist"
                            onclick="event.stopPropagation(); handleAddToPlaylistClick(event, ${index})">
                            +
                        </button>
                        <button class="play-btn" title="Play on YouTube"
                            onclick="event.stopPropagation(); playSong('${songE}', '${artistE}')">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M8 5v14l11-7z"/>
                            </svg>
                        </button>
                    </div>
                </div>
                
                <div class="preview-container hidden"></div>

                <div class="song-tags">
                    <span class="tag">${genreE}</span>
                    <span class="tag emotion">${emotionE}</span>
                </div>
                ${similarity}
            </div>
        </div>
    `;
}

/** Triggered when user clicks '+' on a song card. Reads enriched data from card's dataset. */
function handleAddToPlaylistClick(event, index) {
    event.stopPropagation();
    const card = elements.resultsGrid().querySelector(`[data-song-index="${index}"]`);
    if (!card) return;

    const song = card.dataset.song || '';
    const artist = card.dataset.artist || '';
    const genre = card.dataset.genre || '';
    const emotion = card.dataset.emotion || '';
    const spotifyUrl = card.dataset.spotifyUrl || '';
    const albumArtUrl = card.dataset.albumArt || '';

    showAddToPlaylistPicker(event, song, artist, genre, emotion, spotifyUrl, albumArtUrl);
}

// =============================================================================
// YouTube — open search in new tab (no embed, no modal)
// =============================================================================

/**
 * Opens a YouTube search for the given song + artist in a new browser tab.
 * Uses an anchor element to avoid pop-up blockers.
 */
function playSong(song, artist) {
    const query = encodeURIComponent(`${song} ${artist}`.trim());
    const url = `https://www.youtube.com/results?search_query=${query}`;

    const link = document.createElement('a');
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function closeModal() {
    // Kept for backward-compatibility (modal HTML still in DOM, just never shown now)
    const modal = elements.youtubeModal();
    if (modal) modal.classList.add('hidden');
    const player = elements.youtubePlayer();
    if (player) player.innerHTML = '';
    document.body.style.overflow = '';
}

// =============================================================================
// UI Helpers
// =============================================================================

function showLoading(message) {
    state.isLoading = true;
    const loadingEl = elements.loading();
    loadingEl.classList.remove('hidden');
    // Bug 3 fix: Allow custom loading message (e.g., for CLIP)
    const msgEl = loadingEl.querySelector('p');
    if (msgEl) msgEl.textContent = message || 'Finding perfect songs for you...';
    elements.resultsSection().classList.add('hidden');
}

function hideLoading() {
    state.isLoading = false;
    const loadingEl = elements.loading();
    loadingEl.classList.add('hidden');
    // Reset to default message for next time
    const msgEl = loadingEl.querySelector('p');
    if (msgEl) msgEl.textContent = 'Finding perfect songs for you...';
}

function hideResults() {
    elements.resultsSection().classList.add('hidden');
}

function showToast(message, type = 'info') {
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.style.cssText = `
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%) translateY(100px);
        background: ${type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#8b5cf6'};
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-size: 14px;
        font-weight: 500;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 9999;
        transition: transform 0.3s ease;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.style.transform = 'translateX(-50%) translateY(0)';
    });

    // Remove after delay
    setTimeout(() => {
        toast.style.transform = 'translateX(-50%) translateY(100px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// =============================================================================
// Event Listeners
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Navigation links
    elements.navLinks().forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            switchMode(link.dataset.mode);
        });
    });

    // Setup image upload listeners
    setupImageUploadListeners();

    // Search on Enter key
    elements.searchInput().addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });

    // Artist input Enter key
    elements.artistInput().addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleSearch();
    });

    // Autocomplete
    elements.searchInput().addEventListener('input', handleAutocompleteInput);
    elements.searchInput().addEventListener('focus', () => {
        if (!elements.searchInput().value.trim()) showRecentSearches();
        else handleAutocompleteInput();
    });

    // Re-trigger autocomplete on model change
    document.querySelectorAll('input[name="searchModel"]').forEach(radio => {
        radio.addEventListener('change', () => {
            if (elements.searchInput().value.trim()) handleAutocompleteInput();
        });
    });

    // Close autocomplete on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('#search-mode')) {
            elements.autocompleteDropdown().classList.add('hidden');
        }
        // Close playlist picker on outside click
        const picker = document.getElementById('playlistPicker');
        if (picker && !picker.classList.contains('hidden')) {
            if (!e.target.closest('.add-to-playlist-btn') && !e.target.closest('#playlistPicker')) {
                picker.classList.add('hidden');
            }
        }
    });

    // Escape closes all modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (!document.getElementById('youtubeModal').classList.contains('hidden')) closeModal();
            if (!document.getElementById('authModal').classList.contains('hidden')) closeAuthModal();
            if (!document.getElementById('createPlaylistModal').classList.contains('hidden')) closeCreatePlaylistModal();
            if (!document.getElementById('playlistDetailModal').classList.contains('hidden')) closePlaylistDetailModal();
            document.getElementById('playlistPicker').classList.add('hidden');
        }
    });

    // Login form Enter key
    ['loginUsername', 'loginPassword'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('keypress', e => { if (e.key === 'Enter') handleLogin(); });
    });

    // Register form Enter key
    ['regUsername', 'regEmail', 'regPassword'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('keypress', e => { if (e.key === 'Enter') handleRegister(); });
    });

    // Init auth state
    initAuth();
});


// =============================================================================
// Auth Helpers
// =============================================================================

function authHeaders() {
    return state.token ? { 'Authorization': `Bearer ${state.token}` } : {};
}

async function apiFetch(url, options = {}) {
    const res = await fetch(url, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            ...authHeaders(),
            ...(options.headers || {}),
        },
    });
    return res;
}

function setAuthState(token, user) {
    state.token = token;
    state.user = user;
    localStorage.setItem('authToken', token);

    // Update UI
    document.getElementById('authArea').classList.add('hidden');
    document.getElementById('userArea').classList.remove('hidden');
    document.getElementById('userName').textContent = user.username;
    document.getElementById('userAvatar').textContent = user.username.charAt(0).toUpperCase();

    // Show protected nav links
    document.getElementById('navHistory').style.display = '';
    document.getElementById('navPlaylists').style.display = '';
}

function clearAuthState() {
    state.token = null;
    state.user = null;
    state.playlists = [];
    localStorage.removeItem('authToken');

    document.getElementById('authArea').classList.remove('hidden');
    document.getElementById('userArea').classList.add('hidden');
    document.getElementById('navHistory').style.display = 'none';
    document.getElementById('navPlaylists').style.display = 'none';

    // Redirect to search if on protected tab
    if (state.currentMode === 'history' || state.currentMode === 'playlists') {
        switchMode('search');
    }
}

async function initAuth() {
    if (!state.token) return;
    try {
        const res = await apiFetch('/api/auth/me');
        if (res.ok) {
            const user = await res.json();
            setAuthState(state.token, user);
        } else {
            clearAuthState();
        }
    } catch {
        clearAuthState();
    }
}


// =============================================================================
// Auth Modal
// =============================================================================

function openAuthModal(tab = 'login') {
    document.getElementById('authModal').classList.remove('hidden');
    switchAuthTab(tab);
    document.body.style.overflow = 'hidden';

    // Clear form errors
    document.getElementById('loginError').classList.add('hidden');
    document.getElementById('registerError').classList.add('hidden');
}

function closeAuthModal() {
    document.getElementById('authModal').classList.add('hidden');
    document.body.style.overflow = '';
}

function switchAuthTab(tab) {
    const isLogin = tab === 'login';
    document.getElementById('tabLogin').classList.toggle('active', isLogin);
    document.getElementById('tabRegister').classList.toggle('active', !isLogin);
    document.getElementById('loginForm').classList.toggle('hidden', !isLogin);
    document.getElementById('registerForm').classList.toggle('hidden', isLogin);
}

async function handleLogin() {
    const username = document.getElementById('loginUsername').value.trim();
    const password = document.getElementById('loginPassword').value;
    const errEl = document.getElementById('loginError');
    const btn = document.getElementById('btnLogin');

    if (!username || !password) {
        errEl.textContent = 'Please fill in all fields.';
        errEl.classList.remove('hidden');
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Logging in...';
    errEl.classList.add('hidden');

    try {
        const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password }),
        });
        const data = await res.json();

        if (!res.ok) {
            errEl.textContent = data.detail || 'Login failed.';
            errEl.classList.remove('hidden');
            return;
        }

        setAuthState(data.token, data.user);
        closeAuthModal();
        showToast(`Welcome back, ${data.user.username}! 🎵`, 'info');
        document.getElementById('loginUsername').value = '';
        document.getElementById('loginPassword').value = '';
    } catch (err) {
        errEl.textContent = 'Network error. Please try again.';
        errEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Login';
    }
}

async function handleRegister() {
    const username = document.getElementById('regUsername').value.trim();
    const email = document.getElementById('regEmail').value.trim();
    const password = document.getElementById('regPassword').value;
    const errEl = document.getElementById('registerError');
    const btn = document.getElementById('btnRegister');

    if (!username || !email || !password) {
        errEl.textContent = 'Please fill in all fields.';
        errEl.classList.remove('hidden');
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Creating account...';
    errEl.classList.add('hidden');

    try {
        const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password }),
        });
        const data = await res.json();

        if (!res.ok) {
            // Pydantic validation errors come as array
            const msg = Array.isArray(data.detail)
                ? data.detail.map(e => e.msg).join(', ')
                : (data.detail || 'Registration failed.');
            errEl.textContent = msg;
            errEl.classList.remove('hidden');
            return;
        }

        setAuthState(data.token, data.user);
        closeAuthModal();
        showToast(`Account created! Welcome, ${data.user.username}! 🎉`, 'info');
    } catch (err) {
        errEl.textContent = 'Network error. Please try again.';
        errEl.classList.remove('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Create Account';
    }
}

function handleLogout() {
    clearAuthState();
    showToast('Logged out successfully.', 'info');
}


// =============================================================================
// Search History
// =============================================================================

async function loadHistory() {
    const container = document.getElementById('historyList');
    if (!state.token) {
        container.innerHTML = '<p class="empty-state">Please login to view your search history.</p>';
        return;
    }

    container.innerHTML = '<p class="empty-state">Loading...</p>';

    try {
        const res = await apiFetch('/api/history?limit=50');
        if (!res.ok) throw new Error('Failed to load history');
        const items = await res.json();

        if (items.length === 0) {
            container.innerHTML = '<p class="empty-state">No searches yet. Start searching songs!</p>';
            return;
        }

        container.innerHTML = items.map(item => {
            const meta = [
                item.query_artist ? `by ${escapeHtml(item.query_artist)}` : '',
                `${item.results_count} results`,
                `${item.model_used === 'model2' ? 'Audio Model' : 'Hybrid Model'}`,
                formatTimeAgo(item.searched_at),
            ].filter(Boolean).join(' · ');

            return `
                <div class="history-item">
                    <span class="history-icon">🔍</span>
                    <div class="history-info">
                        <div class="history-song">${escapeHtml(item.query_song)}</div>
                        <div class="history-meta">${meta}</div>
                    </div>
                    <div class="history-actions">
                        <button class="history-search-again"
                            onclick="searchFromHistory('${escapeHtml(item.query_song)}', '${escapeHtml(item.query_artist || '')}')">
                            Search Again
                        </button>
                        <button class="history-delete" title="Delete" onclick="deleteHistoryItem(${item.id}, this)">✕</button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        container.innerHTML = `<p class="empty-state">Failed to load history: ${escapeHtml(err.message)}</p>`;
    }
}

async function deleteHistoryItem(id, btn) {
    btn.disabled = true;
    try {
        const res = await apiFetch(`/api/history/${id}`, { method: 'DELETE' });
        if (res.ok) {
            btn.closest('.history-item').remove();
            const container = document.getElementById('historyList');
            if (!container.querySelector('.history-item')) {
                container.innerHTML = '<p class="empty-state">No searches yet.</p>';
            }
        }
    } catch { btn.disabled = false; }
}

async function clearAllHistory() {
    if (!confirm('Clear all search history?')) return;
    try {
        const res = await apiFetch('/api/history', { method: 'DELETE' });
        if (res.ok) {
            document.getElementById('historyList').innerHTML = '<p class="empty-state">No searches yet.</p>';
            showToast('History cleared!', 'info');
        }
    } catch (err) {
        showToast('Failed to clear history.', 'error');
    }
}

function searchFromHistory(song, artist) {
    switchMode('search');
    elements.searchInput().value = song;
    elements.artistInput().value = artist || '';
    handleSearch();
}

function formatTimeAgo(isoString) {
    const date = new Date(isoString);
    const diff = (Date.now() - date.getTime()) / 1000;
    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}


// =============================================================================
// Playlists
// =============================================================================

async function loadPlaylists() {
    const container = document.getElementById('playlistsList');
    if (!state.token) {
        container.innerHTML = '<p class="empty-state">Please login to view your playlists.</p>';
        return;
    }

    container.innerHTML = '<p class="empty-state">Loading...</p>';

    try {
        const res = await apiFetch('/api/playlists');
        if (!res.ok) throw new Error('Failed to load playlists');
        state.playlists = await res.json();

        if (state.playlists.length === 0) {
            container.innerHTML = '<p class="empty-state">No playlists yet. Create your first one!</p>';
            return;
        }

        container.innerHTML = state.playlists.map(p => `
            <div class="playlist-card" onclick="openPlaylistDetail(${p.id})">
                <div class="playlist-card-icon">🎵</div>
                <div class="playlist-card-name">${escapeHtml(p.name)}</div>
                <div class="playlist-card-desc">${escapeHtml(p.description || 'No description')}</div>
                <div class="playlist-card-footer">
                    <span class="playlist-card-count">${p.track_count} track${p.track_count !== 1 ? 's' : ''}</span>
                    <button class="playlist-card-delete" title="Delete playlist"
                        onclick="event.stopPropagation(); deletePlaylist(${p.id}, this)">🗑</button>
                </div>
            </div>
        `).join('');
    } catch (err) {
        container.innerHTML = `<p class="empty-state">Failed to load playlists.</p>`;
    }
}

function openCreatePlaylistModal() {
    if (!state.token) { openAuthModal('login'); return; }
    document.getElementById('newPlaylistName').value = '';
    document.getElementById('newPlaylistDesc').value = '';
    document.getElementById('createPlaylistError').classList.add('hidden');
    document.getElementById('createPlaylistModal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    setTimeout(() => document.getElementById('newPlaylistName').focus(), 100);
}

function closeCreatePlaylistModal() {
    document.getElementById('createPlaylistModal').classList.add('hidden');
    document.body.style.overflow = '';
}

async function handleCreatePlaylist() {
    const name = document.getElementById('newPlaylistName').value.trim();
    const desc = document.getElementById('newPlaylistDesc').value.trim();
    const errEl = document.getElementById('createPlaylistError');

    if (!name) {
        errEl.textContent = 'Please enter a playlist name.';
        errEl.classList.remove('hidden');
        return;
    }

    errEl.classList.add('hidden');

    try {
        const res = await apiFetch('/api/playlists', {
            method: 'POST',
            body: JSON.stringify({ name, description: desc }),
        });
        const data = await res.json();

        if (!res.ok) {
            errEl.textContent = data.detail || 'Failed to create playlist.';
            errEl.classList.remove('hidden');
            return;
        }

        closeCreatePlaylistModal();
        showToast(`Playlist "${name}" created!`, 'info');

        // Refresh if on playlists tab
        if (state.currentMode === 'playlists') loadPlaylists();
    } catch {
        errEl.textContent = 'Network error.';
        errEl.classList.remove('hidden');
    }
}

async function deletePlaylist(id, btn) {
    if (!confirm('Delete this playlist?')) return;
    btn.disabled = true;
    try {
        const res = await apiFetch(`/api/playlists/${id}`, { method: 'DELETE' });
        if (res.ok) {
            loadPlaylists();
            showToast('Playlist deleted.', 'info');
        }
    } catch { btn.disabled = false; }
}

async function openPlaylistDetail(playlistId) {
    document.getElementById('playlistDetailModal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    document.getElementById('playlistDetailName').textContent = 'Loading...';
    document.getElementById('playlistDetailDesc').textContent = '';
    document.getElementById('playlistDetailCount').textContent = '';
    document.getElementById('playlistTrackList').innerHTML = '<p class="empty-state">Loading...</p>';

    try {
        const res = await apiFetch(`/api/playlists/${playlistId}`);
        if (!res.ok) throw new Error('Failed to load playlist');
        const playlist = await res.json();

        document.getElementById('playlistDetailName').textContent = playlist.name;
        document.getElementById('playlistDetailDesc').textContent = playlist.description || '';
        document.getElementById('playlistDetailCount').textContent =
            `${playlist.tracks.length} track${playlist.tracks.length !== 1 ? 's' : ''}`;

        if (playlist.tracks.length === 0) {
            document.getElementById('playlistTrackList').innerHTML =
                '<p class="empty-state">No tracks yet. Add songs from search results!</p>';
            return;
        }

        document.getElementById('playlistTrackList').innerHTML = playlist.tracks.map(t => {
            const artHtml = t.album_art_url
                ? `<img src="${escapeHtml(t.album_art_url)}" alt="art" loading="lazy">`
                : '🎵';
            const spotifyBtn = t.spotify_url
                ? `<a href="${escapeHtml(t.spotify_url)}" target="_blank" rel="noopener" class="btn-spotify-open" onclick="event.stopPropagation()">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/></svg>
                    Spotify
                   </a>`
                : '';
            return `
                <div class="playlist-track-item">
                    <div class="playlist-track-art">${artHtml}</div>
                    <div class="playlist-track-info">
                        <div class="playlist-track-name">${escapeHtml(t.song_name)}</div>
                        <div class="playlist-track-artist">${escapeHtml(t.artist_name)}</div>
                    </div>
                    <div class="playlist-track-actions">
                        ${spotifyBtn}
                        <button class="btn-track-remove" title="Remove"
                            onclick="removeTrackFromPlaylist(${playlistId}, ${t.id}, this)">✕</button>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        document.getElementById('playlistTrackList').innerHTML =
            `<p class="empty-state">Error: ${escapeHtml(err.message)}</p>`;
    }
}

function closePlaylistDetailModal() {
    document.getElementById('playlistDetailModal').classList.add('hidden');
    document.body.style.overflow = '';
    // Refresh playlist list
    if (state.currentMode === 'playlists') loadPlaylists();
}

async function removeTrackFromPlaylist(playlistId, trackId, btn) {
    btn.disabled = true;
    try {
        const res = await apiFetch(`/api/playlists/${playlistId}/tracks/${trackId}`, { method: 'DELETE' });
        if (res.ok) {
            btn.closest('.playlist-track-item').remove();
            // Update count
            const countEl = document.getElementById('playlistDetailCount');
            const current = parseInt(countEl.textContent) || 1;
            countEl.textContent = `${current - 1} track${(current - 1) !== 1 ? 's' : ''}`;
        }
    } catch { btn.disabled = false; }
}


// =============================================================================
// Add to Playlist (from song cards)
// =============================================================================

function showAddToPlaylistPicker(event, song, artist, genre, emotion, spotifyUrl, albumArtUrl) {
    event.stopPropagation();
    if (!state.token) { openAuthModal('login'); return; }

    state.currentPickerSong = { song_name: song, artist_name: artist, genre, emotion, spotify_url: spotifyUrl, album_art_url: albumArtUrl };

    const picker = document.getElementById('playlistPicker');
    const pickerList = document.getElementById('playlistPickerList');

    // Position picker near button
    const rect = event.currentTarget.getBoundingClientRect();
    picker.style.top = `${rect.bottom + window.scrollY + 6}px`;
    picker.style.left = `${Math.min(rect.left, window.innerWidth - 290)}px`;

    // Load playlists into picker
    if (state.playlists.length === 0) {
        pickerList.innerHTML = '<div class="playlist-picker-item" style="color:var(--text-muted)">No playlists yet</div>';
    } else {
        pickerList.innerHTML = state.playlists.map(p => `
            <div class="playlist-picker-item" onclick="addSongToPlaylist(${p.id})">
                🎵 ${escapeHtml(p.name)}
            </div>
        `).join('');
    }

    picker.classList.remove('hidden');

    // Fetch fresh playlist list in background
    apiFetch('/api/playlists').then(r => r.json()).then(playlists => {
        state.playlists = playlists;
        if (playlists.length === 0) {
            pickerList.innerHTML = '<div class="playlist-picker-item" style="color:var(--text-muted)">No playlists yet</div>';
        } else {
            pickerList.innerHTML = playlists.map(p => `
                <div class="playlist-picker-item" onclick="addSongToPlaylist(${p.id})">
                    🎵 ${escapeHtml(p.name)}
                </div>
            `).join('');
        }
    }).catch(() => {});
}

async function addSongToPlaylist(playlistId) {
    document.getElementById('playlistPicker').classList.add('hidden');
    if (!state.currentPickerSong) return;

    try {
        const res = await apiFetch(`/api/playlists/${playlistId}/tracks`, {
            method: 'POST',
            body: JSON.stringify(state.currentPickerSong),
        });

        if (res.ok) {
            const playlist = state.playlists.find(p => p.id === playlistId);
            showToast(`Added to "${playlist ? playlist.name : 'playlist'}"! 🎵`, 'info');
        } else {
            const data = await res.json();
            showToast(data.detail || 'Failed to add track.', 'error');
        }
    } catch {
        showToast('Network error.', 'error');
    }
    state.currentPickerSong = null;
}

// Context-menu: open create playlist from picker
function openCreatePlaylistModalFromPicker() {
    document.getElementById('playlistPicker').classList.add('hidden');
    openCreatePlaylistModal();
}

