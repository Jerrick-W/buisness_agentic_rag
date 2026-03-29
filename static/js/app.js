/**
 * Enterprise AI Assistant — Frontend Logic
 */
(function () {
    'use strict';

    // DOM elements
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const themeToggle = document.getElementById('themeToggle');
    const newChatBtn = document.getElementById('newChatBtn');
    const sessionList = document.getElementById('sessionList');
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeScreen = document.getElementById('welcomeScreen');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const uploadNotification = document.getElementById('uploadNotification');
    const uploadMessage = document.getElementById('uploadMessage');
    const closeNotification = document.getElementById('closeNotification');
    const kbBtn = document.getElementById('kbBtn');
    const kbOverlay = document.getElementById('kbOverlay');
    const kbClose = document.getElementById('kbClose');
    const kbDocList = document.getElementById('kbDocList');
    const statDocs = document.getElementById('statDocs');
    const statChunks = document.getElementById('statChunks');
    const statDim = document.getElementById('statDim');

    // State
    let currentSessionId = null;
    let sessions = [];
    let isStreaming = false;

    // ---- Theme ----
    function initTheme() {
        const saved = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', saved);
    }

    themeToggle.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    });

    // ---- Sidebar toggle (mobile) ----
    sidebarToggle.addEventListener('click', () => sidebar.classList.toggle('open'));

    // ---- Auto-resize textarea ----
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
        sendBtn.disabled = !messageInput.value.trim();
    });

    // ---- Keyboard: Enter to send, Shift+Enter for newline ----
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (messageInput.value.trim() && !isStreaming) sendMessage();
        }
    });

    sendBtn.addEventListener('click', () => {
        if (messageInput.value.trim() && !isStreaming) sendMessage();
    });

    // ---- Session management ----
    async function createSession() {
        const res = await fetch('/api/sessions', { method: 'POST' });
        const data = await res.json();
        currentSessionId = data.session_id;
        sessions.unshift({ id: data.session_id, label: '新对话' });
        renderSessions();
        clearMessages();
        welcomeScreen.style.display = 'flex';
        messageInput.focus();
    }

    function renderSessions() {
        sessionList.innerHTML = '';
        sessions.forEach((s) => {
            const el = document.createElement('div');
            el.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
            el.textContent = s.label;
            el.addEventListener('click', () => switchSession(s.id));
            sessionList.appendChild(el);
        });
    }

    async function switchSession(sessionId) {
        currentSessionId = sessionId;
        renderSessions();
        clearMessages();
        // Load history
        const res = await fetch(`/api/sessions/${sessionId}/history`);
        const history = await res.json();
        if (history.length > 0) {
            welcomeScreen.style.display = 'none';
            history.forEach((msg) => appendMessage(msg.role, msg.content, msg.sources));
            scrollToBottom();
        } else {
            welcomeScreen.style.display = 'flex';
        }
    }

    newChatBtn.addEventListener('click', createSession);

    // ---- Messages ----
    function clearMessages() {
        const msgs = messagesContainer.querySelectorAll('.message');
        msgs.forEach((m) => m.remove());
    }

    function appendMessage(role, content, sources) {
        welcomeScreen.style.display = 'none';
        const msgEl = document.createElement('div');
        msgEl.className = `message ${role}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        if (role === 'assistant' && typeof marked !== 'undefined') {
            bubble.innerHTML = marked.parse(content);
        } else {
            bubble.textContent = content;
        }

        if (sources && sources.length > 0) {
            const srcEl = document.createElement('div');
            srcEl.className = 'message-sources';
            srcEl.innerHTML = '📎 来源: ' + sources.map(
                (s) => `<span class="source-tag">${s.doc_name}</span>`
            ).join('');
            bubble.appendChild(srcEl);
        }

        msgEl.appendChild(bubble);
        messagesContainer.appendChild(msgEl);
        return bubble;
    }

    function appendTypingIndicator() {
        welcomeScreen.style.display = 'none';
        const msgEl = document.createElement('div');
        msgEl.className = 'message assistant';
        msgEl.id = 'typingMessage';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

        msgEl.appendChild(bubble);
        messagesContainer.appendChild(msgEl);
        scrollToBottom();
        return bubble;
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    // ---- Send message (SSE streaming) ----
    async function sendMessage() {
        const text = messageInput.value.trim();
        if (!text || isStreaming) return;

        if (!currentSessionId) await createSession();

        // Show user message
        appendMessage('user', text);
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendBtn.disabled = true;
        scrollToBottom();

        // Update session label
        const session = sessions.find((s) => s.id === currentSessionId);
        if (session && session.label === '新对话') {
            session.label = text.substring(0, 30) + (text.length > 30 ? '...' : '');
            renderSessions();
        }

        // Show typing indicator
        const bubble = appendTypingIndicator();
        isStreaming = true;

        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, message: text }),
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = '';
            let sources = [];
            bubble.innerHTML = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (!line.startsWith('data:')) continue;
                    const payload = line.substring(5).trim();
                    if (!payload) continue;

                    try {
                        // SSE may wrap in another data layer
                        let parsed;
                        try {
                            parsed = JSON.parse(payload);
                        } catch {
                            continue;
                        }

                        // Handle nested SSE format from sse-starlette
                        if (typeof parsed === 'string') {
                            try { parsed = JSON.parse(parsed); } catch { continue; }
                        }

                        if (parsed.type === 'token') {
                            fullText += parsed.content;
                            bubble.textContent = fullText;
                            scrollToBottom();
                        } else if (parsed.type === 'sources') {
                            sources = parsed.content;
                        } else if (parsed.type === 'error') {
                            bubble.textContent = '⚠️ ' + parsed.content;
                        }
                    } catch { /* skip unparseable */ }
                }
            }

            // Render final markdown and add sources
            if (fullText && typeof marked !== 'undefined') {
                bubble.innerHTML = marked.parse(fullText);
            }
            if (sources.length > 0) {
                const srcEl = document.createElement('div');
                srcEl.className = 'message-sources';
                srcEl.innerHTML = '📎 来源: ' + sources.map(
                    (s) => `<span class="source-tag">${s.doc_name}</span>`
                ).join('');
                bubble.appendChild(srcEl);
            }

        } catch (err) {
            bubble.textContent = '⚠️ 请求失败: ' + err.message;
        } finally {
            // Remove typing message wrapper, keep the real message
            const typingMsg = document.getElementById('typingMessage');
            if (typingMsg) typingMsg.removeAttribute('id');
            isStreaming = false;
            sendBtn.disabled = !messageInput.value.trim();
            scrollToBottom();
        }
    }

    // ---- File upload ----
    uploadBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', async () => {
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        showNotification('正在上传 ' + file.name + '...', '');

        try {
            const res = await fetch('/api/documents/upload', {
                method: 'POST',
                body: formData,
            });

            if (res.ok) {
                const data = await res.json();
                showNotification(
                    `✅ "${data.filename}" 上传成功，已生成 ${data.chunk_count} 个分块`,
                    'success'
                );
                if (!kbOverlay.hidden) loadKnowledgeBase();
            } else {
                const err = await res.json();
                showNotification('❌ 上传失败: ' + (err.message || err.detail), 'error');
            }
        } catch (err) {
            showNotification('❌ 上传失败: ' + err.message, 'error');
        }

        fileInput.value = '';
    });

    function showNotification(msg, type) {
        uploadMessage.textContent = msg;
        uploadNotification.hidden = false;
        uploadNotification.className = 'upload-notification' + (type ? ' ' + type : '');
        if (type) {
            setTimeout(() => { uploadNotification.hidden = true; }, 5000);
        }
    }

    closeNotification.addEventListener('click', () => { uploadNotification.hidden = true; });

    // ---- Knowledge Base Panel ----
    kbBtn.addEventListener('click', () => {
        kbOverlay.hidden = false;
        loadKnowledgeBase();
    });

    kbClose.addEventListener('click', () => { kbOverlay.hidden = true; });
    kbOverlay.addEventListener('click', (e) => {
        if (e.target === kbOverlay) kbOverlay.hidden = true;
    });

    async function loadKnowledgeBase() {
        // Load stats
        try {
            const statsRes = await fetch('/api/knowledge-base/stats');
            const stats = await statsRes.json();
            statDocs.textContent = stats.total_documents || 0;
            statChunks.textContent = stats.total_chunks || 0;
            statDim.textContent = stats.vector_dimension || 0;
        } catch { /* ignore */ }

        // Load document list
        try {
            const docsRes = await fetch('/api/documents');
            const docs = await docsRes.json();
            renderDocList(docs);
        } catch {
            kbDocList.innerHTML = '<div class="kb-empty">加载失败</div>';
        }
    }

    function renderDocList(docs) {
        if (!docs || docs.length === 0) {
            kbDocList.innerHTML = '<div class="kb-empty">暂无文档，请上传文件到知识库</div>';
            return;
        }
        kbDocList.innerHTML = '';
        docs.forEach((doc) => {
            const item = document.createElement('div');
            item.className = 'kb-doc-item';

            const sizeStr = doc.file_size > 1024 * 1024
                ? (doc.file_size / 1024 / 1024).toFixed(1) + ' MB'
                : (doc.file_size / 1024).toFixed(1) + ' KB';

            const uploadDate = new Date(doc.upload_time).toLocaleString('zh-CN');
            const statusClass = doc.status || 'ready';

            item.innerHTML = `
                <div class="kb-doc-info">
                    <div class="kb-doc-name">${doc.filename}</div>
                    <div class="kb-doc-meta">
                        <span>${doc.file_type.toUpperCase()}</span>
                        <span>${sizeStr}</span>
                        <span>${doc.chunk_count} 分块</span>
                        <span>${uploadDate}</span>
                        <span class="kb-doc-status ${statusClass}">${statusClass === 'ready' ? '就绪' : statusClass === 'processing' ? '处理中' : '错误'}</span>
                    </div>
                </div>
                <button class="btn-delete-doc" title="删除文档" aria-label="删除文档">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"/>
                        <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                    </svg>
                </button>
            `;

            const deleteBtn = item.querySelector('.btn-delete-doc');
            deleteBtn.addEventListener('click', () => deleteDocument(doc.doc_id, doc.filename));

            kbDocList.appendChild(item);
        });
    }

    async function deleteDocument(docId, filename) {
        if (!confirm(`确定删除文档「${filename}」？此操作不可恢复。`)) return;

        try {
            const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
            if (res.ok) {
                showNotification(`✅ 文档「${filename}」已删除`, 'success');
                loadKnowledgeBase();
            } else {
                const err = await res.json();
                showNotification('❌ 删除失败: ' + (err.message || err.detail), 'error');
            }
        } catch (err) {
            showNotification('❌ 删除失败: ' + err.message, 'error');
        }
    }

    // ---- Init ----
    initTheme();
    createSession();
})();
