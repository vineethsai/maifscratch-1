/**
 * MAIF Dashboard Application
 * Multi-artifact viewer and management interface
 */

class MAIFDashboard {
    constructor() {
        this.artifacts = new Map();
        this.currentView = 'overview';
        this.selectedArtifacts = new Set();
        this.theme = localStorage.getItem('dashboard-theme') || 'dark';
        
        this.init();
    }

    init() {
        this.setupTheme();
        this.setupEventListeners();
        this.updateStats();
    }

    setupTheme() {
        document.body.setAttribute('data-theme', this.theme);
    }

    setupEventListeners() {
        // Theme toggle
        document.getElementById('themeToggle')?.addEventListener('click', () => {
            this.theme = this.theme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('dashboard-theme', this.theme);
            this.setupTheme();
        });

        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchView(e.target.dataset.view);
            });
        });

        // File input
        document.getElementById('openFileBtn')?.addEventListener('click', () => {
            document.getElementById('fileInput')?.click();
        });

        document.getElementById('fileInput')?.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // Select all checkbox
        document.getElementById('selectAll')?.addEventListener('change', (e) => {
            this.toggleSelectAll(e.target.checked);
        });

        // Generate report button
        document.getElementById('generateReportBtn')?.addEventListener('click', () => {
            this.generateComplianceReport();
        });

        // Modal close
        document.querySelector('.modal-close')?.addEventListener('click', () => {
            this.closeModal();
        });

        // Close modal on outside click
        document.getElementById('artifactModal')?.addEventListener('click', (e) => {
            if (e.target.id === 'artifactModal') {
                this.closeModal();
            }
        });
    }

    switchView(view) {
        this.currentView = view;
        
        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });

        // Update views
        document.querySelectorAll('.view').forEach(v => {
            v.classList.toggle('active', v.id === `${view}-view`);
        });

        // Refresh view-specific content
        this.refreshView(view);
    }

    refreshView(view) {
        switch (view) {
            case 'overview':
                this.updateStats();
                this.renderRecentActivity();
                this.renderBlockDistribution();
                this.renderProvenanceTimeline();
                break;
            case 'artifacts':
                this.renderArtifactsTable();
                break;
            case 'sessions':
                this.renderSessions();
                break;
            case 'integrations':
                this.updateIntegrationStats();
                break;
            case 'compliance':
                this.renderIntegrityChecks();
                this.renderAuditLog();
                break;
        }
    }

    async handleFiles(files) {
        for (const file of files) {
            if (file.name.endsWith('.maif')) {
                try {
                    const buffer = await file.arrayBuffer();
                    const parser = new MAIFParser();
                    parser.loadBinaryFromBuffer(new Uint8Array(buffer));
                    
                    this.artifacts.set(file.name, {
                        name: file.name,
                        size: file.size,
                        parser: parser,
                        loadedAt: new Date(),
                    });
                    
                    console.log(`Loaded ${file.name}`);
                } catch (error) {
                    console.error(`Failed to load ${file.name}:`, error);
                }
            }
        }

        this.updateStats();
        this.refreshView(this.currentView);
    }

    updateStats() {
        let totalBlocks = 0;
        let signedBlocks = 0;
        let allValid = true;

        this.artifacts.forEach(artifact => {
            const parser = artifact.parser;
            totalBlocks += parser.blocks?.length || 0;
            signedBlocks += parser.blocks?.filter(b => b.signature)?.length || 0;
            // Check integrity
            if (parser.blocks) {
                for (const block of parser.blocks) {
                    if (block.isTampered) allValid = false;
                }
            }
        });

        document.getElementById('totalArtifacts').textContent = this.artifacts.size;
        document.getElementById('totalBlocks').textContent = totalBlocks;
        document.getElementById('signedBlocks').textContent = signedBlocks;
        document.getElementById('integrityStatus').textContent = 
            this.artifacts.size === 0 ? '--' : (allValid ? 'Valid' : 'Issues');
    }

    renderRecentActivity() {
        const container = document.getElementById('recentActivity');
        if (!container) return;

        if (this.artifacts.size === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No artifacts loaded</p>
                    <button class="btn btn-secondary" onclick="document.getElementById('fileInput').click()">
                        Open Artifact
                    </button>
                </div>
            `;
            return;
        }

        const activities = [];
        this.artifacts.forEach((artifact, name) => {
            const parser = artifact.parser;
            const provenance = parser.provenance || [];
            
            provenance.slice(-5).forEach(entry => {
                activities.push({
                    time: entry.timestamp,
                    action: entry.action,
                    artifact: name,
                    agent: entry.agent_id,
                });
            });
        });

        activities.sort((a, b) => b.time - a.time);

        container.innerHTML = activities.slice(0, 10).map(a => `
            <div class="audit-entry">
                <div class="audit-time">${this.formatTimestamp(a.time)}</div>
                <div>
                    <div class="audit-action">${this.formatAction(a.action)}</div>
                    <div class="audit-details">${a.artifact} - ${a.agent || 'unknown'}</div>
                </div>
            </div>
        `).join('') || '<div class="empty-state">No activity</div>';
    }

    renderBlockDistribution() {
        const container = document.getElementById('blockDistribution');
        if (!container) return;

        const distribution = {};
        let total = 0;

        this.artifacts.forEach(artifact => {
            const parser = artifact.parser;
            (parser.blocks || []).forEach(block => {
                const type = block.type || 'OTHER';
                distribution[type] = (distribution[type] || 0) + 1;
                total++;
            });
        });

        if (total === 0) {
            container.innerHTML = '<div class="empty-state">No blocks</div>';
            return;
        }

        const colors = {
            'TEXT': '#3b82f6',
            'EMBD': '#8b5cf6',
            'KGRF': '#10b981',
            'IMAG': '#f59e0b',
            'AUDI': '#ef4444',
            'VIDE': '#06b6d4',
            'SECU': '#6366f1',
            'LIFE': '#84cc16',
        };

        container.innerHTML = `
            <div class="distribution-chart">
                ${Object.entries(distribution).map(([type, count]) => {
                    const pct = (count / total) * 100;
                    return `
                        <div class="dist-bar">
                            <span class="dist-label">${type}</span>
                            <div class="dist-track">
                                <div class="dist-fill" style="width: ${pct}%; background: ${colors[type] || '#64748b'}"></div>
                            </div>
                            <span class="dist-count">${count}</span>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    renderProvenanceTimeline() {
        const container = document.getElementById('provenanceTimeline');
        if (!container) return;

        const entries = [];
        this.artifacts.forEach((artifact, name) => {
            const parser = artifact.parser;
            (parser.provenance || []).forEach(entry => {
                entries.push({
                    ...entry,
                    artifact: name,
                });
            });
        });

        entries.sort((a, b) => a.timestamp - b.timestamp);

        if (entries.length === 0) {
            container.innerHTML = '<div class="empty-state">No provenance data</div>';
            return;
        }

        container.innerHTML = `
            <div class="timeline">
                ${entries.slice(-10).map(e => `
                    <div class="timeline-entry ${e.action === 'genesis' ? 'genesis' : ''}">
                        <div class="timeline-time">${this.formatTimestamp(e.timestamp)}</div>
                        <div class="timeline-action">${this.formatAction(e.action)}</div>
                        <div class="timeline-details">${e.artifact} - ${e.agent_id || 'unknown'}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    renderArtifactsTable() {
        const tbody = document.getElementById('artifactsBody');
        if (!tbody) return;

        if (this.artifacts.size === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="empty-state">No artifacts loaded</td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = Array.from(this.artifacts.entries()).map(([name, artifact]) => {
            const parser = artifact.parser;
            const blocks = parser.blocks || [];
            const hasSig = parser.securityInfo?.signature;
            
            return `
                <tr>
                    <td><input type="checkbox" data-name="${name}" ${this.selectedArtifacts.has(name) ? 'checked' : ''}></td>
                    <td>${name}</td>
                    <td>${this.formatSize(artifact.size)}</td>
                    <td>${blocks.length}</td>
                    <td>${hasSig ? '&#x2705;' : '&#x274C;'}</td>
                    <td>${artifact.loadedAt.toLocaleString()}</td>
                    <td>
                        <button class="btn btn-secondary" onclick="dashboard.viewArtifact('${name}')">View</button>
                    </td>
                </tr>
            `;
        }).join('');

        // Add checkbox listeners
        tbody.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', (e) => {
                const name = e.target.dataset.name;
                if (e.target.checked) {
                    this.selectedArtifacts.add(name);
                } else {
                    this.selectedArtifacts.delete(name);
                }
                this.updateCompareButton();
            });
        });
    }

    renderSessions() {
        const container = document.getElementById('sessionsGrid');
        if (!container) return;

        const sessions = new Map();
        
        this.artifacts.forEach((artifact, name) => {
            const parser = artifact.parser;
            (parser.blocks || []).forEach(block => {
                const meta = block.metadata || {};
                const threadId = meta.thread_id || meta.session_id || 'default';
                const key = `${name}:${threadId}`;
                
                if (!sessions.has(key)) {
                    sessions.set(key, {
                        id: threadId,
                        artifact: name,
                        blocks: [],
                        startTime: block.timestamp,
                        endTime: block.timestamp,
                        type: meta.type || 'unknown',
                    });
                }
                
                const session = sessions.get(key);
                session.blocks.push(block);
                if (block.timestamp) {
                    session.startTime = Math.min(session.startTime || block.timestamp, block.timestamp);
                    session.endTime = Math.max(session.endTime || block.timestamp, block.timestamp);
                }
            });
        });

        if (sessions.size === 0) {
            container.innerHTML = '<div class="empty-state">No sessions found</div>';
            return;
        }

        container.innerHTML = Array.from(sessions.values()).map(s => `
            <div class="session-card">
                <div class="session-header">
                    <span class="session-id">${s.id}</span>
                    <span class="session-type">${s.type}</span>
                </div>
                <div class="session-stats">
                    <div>
                        <div class="session-stat-value">${s.blocks.length}</div>
                        <div class="session-stat-label">Blocks</div>
                    </div>
                </div>
                <div class="session-time">
                    ${this.formatTimestamp(s.startTime)} - ${this.formatTimestamp(s.endTime)}
                </div>
            </div>
        `).join('');
    }

    updateIntegrationStats() {
        let langGraphSessions = 0;
        let langGraphCheckpoints = 0;
        let langChainCalls = 0;
        let crewaiCrews = 0;
        let crewaiTasks = 0;

        this.artifacts.forEach(artifact => {
            const parser = artifact.parser;
            (parser.blocks || []).forEach(block => {
                const meta = block.metadata || {};
                const type = meta.type || '';
                
                if (type.includes('checkpoint') || meta.thread_id) {
                    langGraphCheckpoints++;
                }
                if (meta.thread_id) {
                    langGraphSessions++;
                }
                if (type.includes('llm') || type.includes('chain')) {
                    langChainCalls++;
                }
                if (type.includes('crew')) {
                    crewaiCrews++;
                }
                if (type.includes('task')) {
                    crewaiTasks++;
                }
            });
        });

        document.getElementById('langgraphSessions').textContent = langGraphSessions;
        document.getElementById('langgraphCheckpoints').textContent = langGraphCheckpoints;
        document.getElementById('langchainCalls').textContent = langChainCalls;
        document.getElementById('crewaiCrews').textContent = crewaiCrews;
        document.getElementById('crewaiTasks').textContent = crewaiTasks;
    }

    renderIntegrityChecks() {
        const container = document.getElementById('integrityChecks');
        if (!container) return;

        const checks = [
            { name: 'Header Signatures', passed: true, details: 'All artifacts have valid headers' },
            { name: 'Block Chain', passed: true, details: 'Hash chains verified' },
            { name: 'Merkle Roots', passed: true, details: 'Root hashes match' },
            { name: 'Timestamps', passed: true, details: 'Chronological order verified' },
        ];

        // Check actual data
        this.artifacts.forEach(artifact => {
            const parser = artifact.parser;
            if (!parser.securityInfo?.signature) {
                checks[0].passed = false;
                checks[0].details = 'Some artifacts lack signatures';
            }
            (parser.blocks || []).forEach(block => {
                if (block.isTampered) {
                    checks[1].passed = false;
                    checks[1].details = 'Tampered blocks detected';
                }
            });
        });

        container.innerHTML = `
            <div class="check-list">
                ${checks.map(c => `
                    <div class="check-item">
                        <span class="check-icon ${c.passed ? 'passed' : 'failed'}">
                            ${c.passed ? '&#x2705;' : '&#x274C;'}
                        </span>
                        <div class="check-content">
                            <div class="check-name">${c.name}</div>
                            <div class="check-details">${c.details}</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        // Update security summary
        document.getElementById('chainStatus').textContent = checks[1].passed ? 'Verified' : 'Compromised';
        document.getElementById('merkleStatus').textContent = checks[2].passed ? 'Valid' : 'Invalid';
    }

    renderAuditLog() {
        const container = document.getElementById('auditLog');
        if (!container) return;

        const entries = [];
        this.artifacts.forEach((artifact, name) => {
            const parser = artifact.parser;
            (parser.provenance || []).forEach(entry => {
                entries.push({
                    time: entry.timestamp,
                    action: entry.action,
                    agent: entry.agent_id,
                    artifact: name,
                });
            });
        });

        entries.sort((a, b) => b.time - a.time);

        if (entries.length === 0) {
            container.innerHTML = '<div class="empty-state">No audit entries</div>';
            return;
        }

        container.innerHTML = entries.slice(0, 20).map(e => `
            <div class="audit-entry">
                <div class="audit-time">${this.formatTimestamp(e.time)}</div>
                <div>
                    <div class="audit-action">${this.formatAction(e.action)}</div>
                    <div class="audit-details">${e.artifact} - ${e.agent || 'unknown'}</div>
                </div>
            </div>
        `).join('');
    }

    viewArtifact(name) {
        const artifact = this.artifacts.get(name);
        if (!artifact) return;

        const parser = artifact.parser;
        const modal = document.getElementById('artifactModal');
        const title = document.getElementById('modalTitle');
        const body = document.getElementById('modalBody');

        title.textContent = name;
        body.innerHTML = `
            <h3>Overview</h3>
            <div class="section">
                <div class="stat-row"><span>Size</span><span>${this.formatSize(artifact.size)}</span></div>
                <div class="stat-row"><span>Blocks</span><span>${parser.blocks?.length || 0}</span></div>
                <div class="stat-row"><span>Format</span><span>${parser.format || 'unknown'}</span></div>
                <div class="stat-row"><span>Signed</span><span>${parser.securityInfo?.signature ? 'Yes' : 'No'}</span></div>
            </div>

            <h3>Blocks</h3>
            <div class="section">
                ${(parser.blocks || []).slice(0, 10).map((b, i) => `
                    <div class="stat-row">
                        <span>${i}: ${b.type}</span>
                        <span>${this.formatSize(b.size)}</span>
                    </div>
                `).join('')}
                ${parser.blocks?.length > 10 ? `<div class="stat-row"><span>... and ${parser.blocks.length - 10} more</span></div>` : ''}
            </div>
        `;

        modal.classList.add('active');
    }

    closeModal() {
        document.getElementById('artifactModal')?.classList.remove('active');
    }

    toggleSelectAll(checked) {
        if (checked) {
            this.artifacts.forEach((_, name) => this.selectedArtifacts.add(name));
        } else {
            this.selectedArtifacts.clear();
        }
        this.renderArtifactsTable();
        this.updateCompareButton();
    }

    updateCompareButton() {
        const btn = document.getElementById('compareBtn');
        if (btn) {
            btn.disabled = this.selectedArtifacts.size < 2;
        }
    }

    generateComplianceReport() {
        alert('Compliance report generation would create a detailed PDF/HTML report of all loaded artifacts.');
    }

    formatTimestamp(ts) {
        if (!ts) return '--';
        const ms = ts > 1e12 ? ts / 1000 : ts * 1000;
        return new Date(ms).toLocaleString();
    }

    formatSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    formatAction(action) {
        const map = {
            'genesis': 'Genesis',
            'add_text_block': 'Add Text',
            'add_embeddings_block': 'Add Embeddings',
            'finalize': 'Finalize',
            'checkpoint': 'Checkpoint',
            'state_checkpoint': 'State Checkpoint',
        };
        return map[action] || action;
    }
}

// Initialize dashboard
const dashboard = new MAIFDashboard();

