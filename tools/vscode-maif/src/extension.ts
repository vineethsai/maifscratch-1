import * as vscode from 'vscode';
import { MAIFBinaryViewerProvider } from './binaryViewer';
import { MAIFOverviewProvider, MAIFBlocksProvider, MAIFProvenanceProvider, MAIFSessionsProvider, MAIFIntegrityProvider } from './treeProviders';
import { MAIFParser } from './parser';

let parser: MAIFParser | undefined;
let overviewProvider: MAIFOverviewProvider | undefined;
let blocksProvider: MAIFBlocksProvider | undefined;
let provenanceProvider: MAIFProvenanceProvider | undefined;
let sessionsProvider: MAIFSessionsProvider | undefined;
let integrityProvider: MAIFIntegrityProvider | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('MAIF Explorer extension is now active');

    // Initialize parser
    parser = new MAIFParser();

    // Register custom editor for .maif files
    const binaryViewerProvider = new MAIFBinaryViewerProvider(context);
    context.subscriptions.push(
        vscode.window.registerCustomEditorProvider(
            'maif.binaryViewer',
            binaryViewerProvider,
            {
                webviewOptions: { retainContextWhenHidden: true },
                supportsMultipleEditorsPerDocument: false
            }
        )
    );

    // Register tree view providers
    overviewProvider = new MAIFOverviewProvider();
    blocksProvider = new MAIFBlocksProvider();
    provenanceProvider = new MAIFProvenanceProvider();
    sessionsProvider = new MAIFSessionsProvider();
    integrityProvider = new MAIFIntegrityProvider();

    context.subscriptions.push(
        vscode.window.registerTreeDataProvider('maifOverview', overviewProvider),
        vscode.window.registerTreeDataProvider('maifBlocks', blocksProvider),
        vscode.window.registerTreeDataProvider('maifProvenance', provenanceProvider),
        vscode.window.registerTreeDataProvider('maifSessions', sessionsProvider),
        vscode.window.registerTreeDataProvider('maifIntegrity', integrityProvider)
    );

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('maif.openExplorer', async (uri?: vscode.Uri) => {
            if (!uri) {
                const uris = await vscode.window.showOpenDialog({
                    canSelectMany: false,
                    filters: {
                        'MAIF Files': ['maif', 'json']
                    }
                });
                if (uris && uris.length > 0) {
                    uri = uris[0];
                }
            }

            if (uri) {
                await openMAIFFile(uri);
            }
        }),

        vscode.commands.registerCommand('maif.showProvenance', async () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && parser?.manifest) {
                const provenance = parser.manifest.signature_metadata?.provenance_chain || [];
                const panel = vscode.window.createWebviewPanel(
                    'maifProvenance',
                    'MAIF Provenance Chain',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );
                panel.webview.html = getProvenanceHtml(provenance);
            } else {
                vscode.window.showWarningMessage('No MAIF manifest loaded');
            }
        }),

        vscode.commands.registerCommand('maif.verifySignature', async () => {
            if (parser?.manifest?.signature) {
                vscode.window.showInformationMessage(
                    `Signature present from: ${parser.manifest.signature_metadata?.signer_id || 'Unknown'}`
                );
            } else {
                vscode.window.showWarningMessage('No signature found in manifest');
            }
        }),

        vscode.commands.registerCommand('maif.showHexView', async () => {
            if (parser?.binaryData) {
                const panel = vscode.window.createWebviewPanel(
                    'maifHexView',
                    'MAIF Hex View',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );
                panel.webview.html = getHexViewHtml(parser.binaryData);
            } else {
                vscode.window.showWarningMessage('No binary data loaded');
            }
        }),

        vscode.commands.registerCommand('maif.selectBlock', async (blockIndex: number) => {
            if (parser?.blocks && blockIndex >= 0 && blockIndex < parser.blocks.length) {
                const block = parser.blocks[blockIndex];
                const panel = vscode.window.createWebviewPanel(
                    'maifBlockDetail',
                    `Block: ${block.type}`,
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );
                panel.webview.html = getBlockDetailHtml(block);
            }
        }),

        // New commands for enhanced functionality
        vscode.commands.registerCommand('maif.verifyIntegrity', async () => {
            if (!parser?.blocks) {
                vscode.window.showWarningMessage('No MAIF file loaded');
                return;
            }

            const results = verifyArtifactIntegrity(parser);
            const panel = vscode.window.createWebviewPanel(
                'maifIntegrity',
                'MAIF Integrity Verification',
                vscode.ViewColumn.Beside,
                { enableScripts: true }
            );
            panel.webview.html = getIntegrityReportHtml(results);

            if (results.isValid) {
                vscode.window.showInformationMessage('MAIF artifact integrity verified successfully');
            } else {
                vscode.window.showErrorMessage(`Integrity verification failed: ${results.errors.length} issues found`);
            }
        }),

        vscode.commands.registerCommand('maif.exportToJson', async () => {
            if (!parser?.blocks) {
                vscode.window.showWarningMessage('No MAIF file loaded');
                return;
            }

            const exportData = {
                fileInfo: parser.fileInfo,
                manifest: parser.manifest,
                blocks: parser.blocks.map(b => ({
                    type: b.type,
                    size: b.size,
                    hash: b.hash,
                    metadata: b.metadata
                })),
                provenance: parser.provenance,
                exportedAt: new Date().toISOString()
            };

            const uri = await vscode.window.showSaveDialog({
                defaultUri: vscode.Uri.file('maif_export.json'),
                filters: { 'JSON': ['json'] }
            });

            if (uri) {
                await vscode.workspace.fs.writeFile(uri, Buffer.from(JSON.stringify(exportData, null, 2)));
                vscode.window.showInformationMessage(`Exported to ${uri.fsPath}`);
            }
        }),

        vscode.commands.registerCommand('maif.showTimeline', async () => {
            if (!parser?.blocks) {
                vscode.window.showWarningMessage('No MAIF file loaded');
                return;
            }

            const panel = vscode.window.createWebviewPanel(
                'maifTimeline',
                'MAIF Session Timeline',
                vscode.ViewColumn.Beside,
                { enableScripts: true }
            );
            panel.webview.html = getTimelineHtml(parser.blocks, parser.provenance || []);
        }),

        vscode.commands.registerCommand('maif.searchBlocks', async () => {
            if (!parser?.blocks) {
                vscode.window.showWarningMessage('No MAIF file loaded');
                return;
            }

            const query = await vscode.window.showInputBox({
                prompt: 'Search blocks by type, content, or metadata',
                placeHolder: 'e.g., TEXT, checkpoint, error'
            });

            if (query) {
                const results = searchBlocks(parser.blocks, query);
                if (results.length > 0) {
                    const panel = vscode.window.createWebviewPanel(
                        'maifSearch',
                        `Search Results: ${query}`,
                        vscode.ViewColumn.Beside,
                        { enableScripts: true }
                    );
                    panel.webview.html = getSearchResultsHtml(results, query);
                } else {
                    vscode.window.showInformationMessage(`No blocks found matching "${query}"`);
                }
            }
        }),

        vscode.commands.registerCommand('maif.generateReport', async () => {
            if (!parser?.blocks) {
                vscode.window.showWarningMessage('No MAIF file loaded');
                return;
            }

            const panel = vscode.window.createWebviewPanel(
                'maifReport',
                'MAIF Audit Report',
                vscode.ViewColumn.Beside,
                { enableScripts: true }
            );
            panel.webview.html = getAuditReportHtml(parser);
        }),

        vscode.commands.registerCommand('maif.compareCheckpoints', async () => {
            vscode.window.showInformationMessage('Select two checkpoint blocks to compare');
            // Implementation would require user selection of two blocks
        })
    );

    // Watch for document changes
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(async (doc) => {
            if (doc.fileName.endsWith('_manifest.json') || doc.fileName.endsWith('-manifest.json')) {
                await loadManifest(doc.uri);
            }
        })
    );
}

async function openMAIFFile(uri: vscode.Uri): Promise<void> {
    try {
        if (uri.fsPath.endsWith('.maif')) {
            // Load binary first - it will auto-detect format
            await loadBinary(uri);
            
            // If it's legacy format (not secure), try to find accompanying manifest
            if (parser?.format !== 'secure') {
                const manifestPath = uri.fsPath.replace('.maif', '_manifest.json');
                const manifestUri = vscode.Uri.file(manifestPath);
                
                try {
                    await vscode.workspace.fs.stat(manifestUri);
                    await loadManifest(manifestUri);
                } catch {
                    // Manifest not found, continue with binary-only view
                }
            }

            await vscode.commands.executeCommand('vscode.openWith', uri, 'maif.binaryViewer');
            
            // Show format info
            const formatMsg = parser?.format === 'secure' 
                ? ' Secure MAIF format (self-contained with embedded security)'
                : '📁 Legacy MAIF format';
            vscode.window.setStatusBarMessage(formatMsg, 5000);
        } else if (uri.fsPath.endsWith('.json')) {
            await loadManifest(uri);
            await vscode.window.showTextDocument(uri);
        }

        refreshTreeViews();
    } catch (error) {
        vscode.window.showErrorMessage(`Failed to open MAIF file: ${error}`);
    }
}

async function loadManifest(uri: vscode.Uri): Promise<void> {
    const content = await vscode.workspace.fs.readFile(uri);
    const text = new TextDecoder().decode(content);
    parser?.loadManifestFromString(text);
}

async function loadBinary(uri: vscode.Uri): Promise<void> {
    const content = await vscode.workspace.fs.readFile(uri);
    parser?.loadBinaryFromBuffer(content);
}

function refreshTreeViews(): void {
    if (parser?.manifest) {
        overviewProvider?.setManifest(parser.manifest, parser.fileInfo || undefined, parser.securityInfo || undefined);
        blocksProvider?.setBlocks(parser.blocks);
        provenanceProvider?.setProvenance(parser.provenance || []);
        sessionsProvider?.setSessions(extractSessions(parser.blocks));
        integrityProvider?.setIntegrity(verifyArtifactIntegrity(parser));
    }
}

// Helper functions for new features
interface IntegrityResult {
    isValid: boolean;
    errors: string[];
    warnings: string[];
    checks: { name: string; passed: boolean; details: string }[];
}

function verifyArtifactIntegrity(parser: MAIFParser): IntegrityResult {
    const result: IntegrityResult = {
        isValid: true,
        errors: [],
        warnings: [],
        checks: []
    };

    // Check header signature
    const headerCheck = {
        name: 'Header Signature',
        passed: !!parser.securityInfo?.signature,
        details: parser.securityInfo?.signature ? 'Ed25519 signature present' : 'No signature found'
    };
    result.checks.push(headerCheck);

    // Check merkle root
    const merkleCheck = {
        name: 'Merkle Root',
        passed: !!parser.securityInfo?.merkleRoot && parser.securityInfo.merkleRoot !== '0'.repeat(64),
        details: parser.securityInfo?.merkleRoot ? `Root: ${parser.securityInfo.merkleRoot.slice(0, 16)}...` : 'No merkle root'
    };
    result.checks.push(merkleCheck);

    // Check block chain
    let chainValid = true;
    let prevHash = '0'.repeat(64);
    for (const block of parser.blocks || []) {
        if (block.previous_hash && block.previous_hash !== prevHash) {
            chainValid = false;
            result.errors.push(`Block chain broken at offset ${block.offset}`);
        }
        prevHash = block.hash || prevHash;
    }
    result.checks.push({
        name: 'Block Chain',
        passed: chainValid,
        details: chainValid ? `${parser.blocks?.length || 0} blocks verified` : 'Chain integrity compromised'
    });

    // Check block signatures
    const signedBlocks = parser.blocks?.filter(b => b.signature) || [];
    result.checks.push({
        name: 'Block Signatures',
        passed: true,
        details: `${signedBlocks.length}/${parser.blocks?.length || 0} blocks signed`
    });

    result.isValid = result.errors.length === 0;
    return result;
}

interface Session {
    id: string;
    startTime: number;
    endTime: number;
    blockCount: number;
    type: string;
}

function extractSessions(blocks: any[]): Session[] {
    const sessions: Map<string, Session> = new Map();

    for (const block of blocks || []) {
        const meta = block.metadata || {};
        const threadId = meta.thread_id || meta.session_id || 'default';

        if (!sessions.has(threadId)) {
            sessions.set(threadId, {
                id: threadId,
                startTime: block.timestamp || 0,
                endTime: block.timestamp || 0,
                blockCount: 0,
                type: meta.type || 'unknown'
            });
        }

        const session = sessions.get(threadId)!;
        session.blockCount++;
        if (block.timestamp) {
            session.endTime = Math.max(session.endTime, block.timestamp);
        }
    }

    return Array.from(sessions.values());
}

function searchBlocks(blocks: any[], query: string): any[] {
    const queryLower = query.toLowerCase();
    return blocks.filter(block => {
        // Search in type
        if (block.type?.toLowerCase().includes(queryLower)) return true;
        // Search in metadata
        const metaStr = JSON.stringify(block.metadata || {}).toLowerCase();
        if (metaStr.includes(queryLower)) return true;
        // Search in content (if text)
        if (block.content?.toLowerCase().includes(queryLower)) return true;
        return false;
    });
}

function getProvenanceHtml(provenance: any[]): string {
    const entries = provenance.map((entry, i) => `
        <div class="entry ${entry.action === 'genesis' ? 'genesis' : ''}">
            <div class="marker"></div>
            <div class="content">
                <div class="action">${formatAction(entry.action)}</div>
                <div class="time">${formatTimestamp(entry.timestamp)}</div>
                <div class="details">
                    <div><strong>Agent:</strong> ${entry.agent_id}</div>
                    <div><strong>DID:</strong> ${entry.agent_did || 'N/A'}</div>
                    <div><strong>Entry Hash:</strong> <code>${truncate(entry.entry_hash)}</code></div>
                    <div><strong>Block Hash:</strong> <code>${truncate(entry.block_hash)}</code></div>
                </div>
                ${entry.signature ? `<div class="signature"><strong>Signature:</strong> <code>${truncate(entry.signature, 50)}</code></div>` : ''}
            </div>
        </div>
    `).join('');

    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 18px; margin-bottom: 20px; }
            .timeline { position: relative; padding-left: 30px; }
            .timeline::before { content: ''; position: absolute; left: 10px; top: 0; bottom: 0; width: 2px; background: var(--vscode-textSeparator-foreground); }
            .entry { position: relative; margin-bottom: 20px; padding: 15px; background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 8px; }
            .entry.genesis { border-left: 3px solid var(--vscode-charts-green); }
            .marker { position: absolute; left: -26px; top: 20px; width: 12px; height: 12px; border-radius: 50%; background: var(--vscode-button-background); }
            .genesis .marker { background: var(--vscode-charts-green); }
            .action { font-weight: bold; color: var(--vscode-textLink-foreground); margin-bottom: 5px; }
            .time { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 10px; }
            .details { font-size: 13px; line-height: 1.6; }
            .signature { margin-top: 10px; font-size: 12px; padding-top: 10px; border-top: 1px solid var(--vscode-textSeparator-foreground); }
            code { background: var(--vscode-textCodeBlock-background); padding: 2px 6px; border-radius: 3px; font-size: 11px; }
        </style>
    </head>
    <body>
        <h1> Provenance Chain (${provenance.length} entries)</h1>
        <div class="timeline">${entries}</div>
    </body>
    </html>`;
}

function getHexViewHtml(data: Uint8Array): string {
    const lines: string[] = [];
    for (let i = 0; i < Math.min(data.length, 2048); i += 16) {
        const offset = i.toString(16).padStart(8, '0');
        const bytes = Array.from(data.slice(i, Math.min(i + 16, data.length)));
        const hex = bytes.map(b => b.toString(16).padStart(2, '0')).join(' ').padEnd(47, ' ');
        const ascii = bytes.map(b => (b >= 32 && b < 127) ? String.fromCharCode(b) : '.').join('');
        lines.push(`<div class="line"><span class="offset">${offset}</span><span class="hex">${hex}</span><span class="ascii">${ascii}</span></div>`);
    }

    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-editor-font-family); font-size: 13px; padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 16px; margin-bottom: 15px; }
            .info { font-size: 12px; color: var(--vscode-descriptionForeground); margin-bottom: 20px; }
            .hex-view { background: var(--vscode-editor-background); padding: 15px; border-radius: 8px; overflow: auto; }
            .line { display: flex; gap: 20px; line-height: 1.8; }
            .offset { color: var(--vscode-descriptionForeground); min-width: 80px; }
            .hex { color: var(--vscode-textLink-foreground); min-width: 360px; }
            .ascii { color: var(--vscode-foreground); }
        </style>
    </head>
    <body>
        <h1>🔢 Hex View</h1>
        <div class="info">${data.length.toLocaleString()} bytes total (showing first 2KB)</div>
        <div class="hex-view">${lines.join('')}</div>
    </body>
    </html>`;
}

function getBlockDetailHtml(block: any): string {
    const statusBadges: string[] = [];
    if (block.isSigned) statusBadges.push('<span class="status-badge signed"> Signed</span>');
    if (block.isImmutable) statusBadges.push('<span class="status-badge immutable"> Immutable</span>');
    if (block.isTampered) statusBadges.push('<span class="status-badge tampered"> TAMPERED</span>');

    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 18px; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
            .badge { background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 4px 8px; border-radius: 4px; font-size: 12px; }
            .status-badges { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
            .status-badge { padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; }
            .status-badge.signed { background: rgba(16, 185, 129, 0.15); color: #10b981; }
            .status-badge.immutable { background: rgba(99, 102, 241, 0.15); color: #6366f1; }
            .status-badge.tampered { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
            .section { margin-bottom: 20px; }
            .section-title { font-size: 12px; text-transform: uppercase; color: var(--vscode-descriptionForeground); margin-bottom: 10px; }
            .row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--vscode-textSeparator-foreground); }
            .label { color: var(--vscode-descriptionForeground); }
            .value { font-family: var(--vscode-editor-font-family); }
            .hash { font-family: var(--vscode-editor-font-family); font-size: 11px; background: var(--vscode-textCodeBlock-background); padding: 10px; border-radius: 6px; word-break: break-all; }
            pre { background: var(--vscode-textCodeBlock-background); padding: 15px; border-radius: 6px; overflow: auto; font-size: 12px; }
        </style>
    </head>
    <body>
        <h1><span class="badge">${block.type}</span> Block Details</h1>
        
        ${statusBadges.length > 0 ? `<div class="status-badges">${statusBadges.join('')}</div>` : ''}
        
        <div class="section">
            <div class="section-title">Information</div>
            <div class="row"><span class="label">Block ID</span><span class="value">${block.block_id || 'N/A'}</span></div>
            <div class="row"><span class="label">Offset</span><span class="value">${block.offset} (0x${block.offset?.toString(16) || '0'})</span></div>
            <div class="row"><span class="label">Size</span><span class="value">${formatSize(block.size)} (${block.size} bytes)</span></div>
            ${block.dataSize !== undefined ? `<div class="row"><span class="label">Data Size</span><span class="value">${formatSize(block.dataSize)} (${block.dataSize} bytes)</span></div>` : ''}
            <div class="row"><span class="label">Version</span><span class="value">${block.version || 1}</span></div>
            ${block.timestamp ? `<div class="row"><span class="label">Timestamp</span><span class="value">${formatTimestamp(block.timestamp)}</span></div>` : ''}
        </div>

        <div class="section">
            <div class="section-title">Content Hash</div>
            <div class="hash">${block.hash || 'N/A'}</div>
        </div>

        ${block.previous_hash && block.previous_hash !== '0'.repeat(64) ? `
        <div class="section">
            <div class="section-title">Previous Block Hash (Chain Link)</div>
            <div class="hash">${block.previous_hash}</div>
        </div>
        ` : ''}

        ${block.signature ? `
        <div class="section">
            <div class="section-title">Block Signature (RSA-PSS)</div>
            <div class="hash" style="font-size: 9px;">${block.signature}</div>
        </div>
        ` : ''}

        ${block.metadata ? `
        <div class="section">
            <div class="section-title">Metadata</div>
            <pre>${JSON.stringify(block.metadata, null, 2)}</pre>
        </div>
        ` : ''}
    </body>
    </html>`;
}

function formatAction(action: string): string {
    const map: Record<string, string> = {
        'genesis': '🌟 Genesis',
        'add_text_block': ' Add Text Block',
        'add_embeddings_block': ' Add Embeddings',
        'add_knowledge_graph': ' Add Knowledge Graph',
        'add_image_block': ' Add Image',
        'add_audio_block': ' Add Audio',
        'add_video_block': ' Add Video',
        'finalize': ' Finalize',
        'sign': '✍️ Sign',
        'verify': ' Verify'
    };
    return map[action] || action;
}

function formatTimestamp(ts: number): string {
    // Handle microseconds (secure format) vs seconds (legacy)
    const ms = ts > 1e12 ? ts / 1000 : ts * 1000;
    return new Date(ms).toLocaleString();
}

function truncate(str: string | undefined, len = 20): string {
    if (!str) return 'N/A';
    return str.length > len * 2 ? `${str.slice(0, len)}...${str.slice(-len)}` : str;
}

function formatSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getIntegrityReportHtml(results: IntegrityResult): string {
    const statusIcon = results.isValid ? '&#x2705;' : '&#x274C;';
    const checksHtml = results.checks.map(check => `
        <div class="check ${check.passed ? 'passed' : 'failed'}">
            <span class="check-icon">${check.passed ? '&#x2705;' : '&#x274C;'}</span>
            <div class="check-content">
                <div class="check-name">${check.name}</div>
                <div class="check-details">${check.details}</div>
            </div>
        </div>
    `).join('');

    const errorsHtml = results.errors.length > 0 
        ? `<div class="errors"><h3>Errors</h3>${results.errors.map(e => `<div class="error">&#x274C; ${e}</div>`).join('')}</div>`
        : '';

    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
            .header { display: flex; align-items: center; gap: 15px; margin-bottom: 25px; }
            .status { font-size: 24px; }
            h1 { font-size: 20px; margin: 0; }
            .subtitle { color: var(--vscode-descriptionForeground); font-size: 13px; }
            .checks { display: flex; flex-direction: column; gap: 12px; }
            .check { display: flex; align-items: center; gap: 12px; padding: 12px; background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 8px; }
            .check.failed { border-left: 3px solid #ef4444; }
            .check.passed { border-left: 3px solid #10b981; }
            .check-icon { font-size: 18px; }
            .check-name { font-weight: 500; }
            .check-details { font-size: 12px; color: var(--vscode-descriptionForeground); margin-top: 4px; }
            .errors { margin-top: 20px; padding: 15px; background: rgba(239,68,68,0.1); border-radius: 8px; }
            .error { padding: 5px 0; }
        </style>
    </head>
    <body>
        <div class="header">
            <span class="status">${statusIcon}</span>
            <div>
                <h1>Integrity Verification</h1>
                <div class="subtitle">${results.isValid ? 'All checks passed' : 'Issues detected'}</div>
            </div>
        </div>
        <div class="checks">${checksHtml}</div>
        ${errorsHtml}
    </body>
    </html>`;
}

function getTimelineHtml(blocks: any[], provenance: any[]): string {
    const events: { time: number; type: string; details: string }[] = [];

    for (const block of blocks) {
        if (block.timestamp) {
            const meta = block.metadata || {};
            events.push({
                time: block.timestamp,
                type: meta.type || block.type || 'block',
                details: meta.thread_id ? `Thread: ${meta.thread_id}` : ''
            });
        }
    }

    for (const entry of provenance) {
        if (entry.timestamp) {
            events.push({
                time: entry.timestamp,
                type: entry.action || 'provenance',
                details: entry.agent_id || ''
            });
        }
    }

    events.sort((a, b) => a.time - b.time);

    const timelineHtml = events.map(e => `
        <div class="event">
            <div class="time">${formatTimestamp(e.time)}</div>
            <div class="type">${e.type}</div>
            <div class="details">${e.details}</div>
        </div>
    `).join('');

    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 18px; margin-bottom: 20px; }
            .timeline { border-left: 2px solid var(--vscode-textSeparator-foreground); padding-left: 20px; }
            .event { position: relative; padding: 10px 0; border-bottom: 1px solid var(--vscode-textSeparator-foreground); }
            .event::before { content: ''; position: absolute; left: -26px; top: 15px; width: 10px; height: 10px; border-radius: 50%; background: var(--vscode-button-background); }
            .time { font-size: 11px; color: var(--vscode-descriptionForeground); }
            .type { font-weight: 500; color: var(--vscode-textLink-foreground); }
            .details { font-size: 12px; color: var(--vscode-descriptionForeground); }
        </style>
    </head>
    <body>
        <h1>Session Timeline (${events.length} events)</h1>
        <div class="timeline">${timelineHtml}</div>
    </body>
    </html>`;
}

function getSearchResultsHtml(results: any[], query: string): string {
    const resultsHtml = results.map((block, i) => `
        <div class="result">
            <div class="result-header">
                <span class="type-badge">${block.type}</span>
                <span class="offset">Offset: ${block.offset}</span>
            </div>
            <div class="result-content">
                ${block.metadata ? `<pre>${JSON.stringify(block.metadata, null, 2)}</pre>` : 'No metadata'}
            </div>
        </div>
    `).join('');

    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
            h1 { font-size: 18px; margin-bottom: 5px; }
            .subtitle { color: var(--vscode-descriptionForeground); margin-bottom: 20px; }
            .result { padding: 15px; margin-bottom: 10px; background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 8px; }
            .result-header { display: flex; justify-content: space-between; margin-bottom: 10px; }
            .type-badge { background: var(--vscode-badge-background); color: var(--vscode-badge-foreground); padding: 3px 8px; border-radius: 4px; font-size: 11px; }
            .offset { font-size: 11px; color: var(--vscode-descriptionForeground); }
            pre { font-size: 11px; overflow: auto; background: var(--vscode-textCodeBlock-background); padding: 10px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <h1>Search Results</h1>
        <div class="subtitle">${results.length} blocks matching "${query}"</div>
        ${resultsHtml}
    </body>
    </html>`;
}

function getAuditReportHtml(parser: MAIFParser): string {
    const blocks = parser.blocks || [];
    const provenance = parser.provenance || [];
    const fileInfo = parser.fileInfo || {};

    const blockTypes: Record<string, number> = {};
    for (const block of blocks) {
        const type = block.type || 'unknown';
        blockTypes[type] = (blockTypes[type] || 0) + 1;
    }

    const blockStatsHtml = Object.entries(blockTypes).map(([type, count]) => 
        `<div class="stat-row"><span>${type}</span><span>${count}</span></div>`
    ).join('');

    return `<!DOCTYPE html>
    <html>
    <head>
        <style>
            body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); max-width: 800px; }
            h1 { font-size: 20px; border-bottom: 2px solid var(--vscode-textSeparator-foreground); padding-bottom: 10px; }
            h2 { font-size: 16px; margin-top: 25px; color: var(--vscode-textLink-foreground); }
            .section { background: var(--vscode-editor-inactiveSelectionBackground); padding: 15px; border-radius: 8px; margin: 10px 0; }
            .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--vscode-textSeparator-foreground); }
            .stat-row:last-child { border-bottom: none; }
            .summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
            .summary-card { background: var(--vscode-editor-inactiveSelectionBackground); padding: 15px; border-radius: 8px; text-align: center; }
            .summary-value { font-size: 24px; font-weight: bold; color: var(--vscode-textLink-foreground); }
            .summary-label { font-size: 12px; color: var(--vscode-descriptionForeground); }
            .footer { margin-top: 30px; padding-top: 15px; border-top: 1px solid var(--vscode-textSeparator-foreground); font-size: 11px; color: var(--vscode-descriptionForeground); }
        </style>
    </head>
    <body>
        <h1>MAIF Audit Report</h1>
        <div class="footer">Generated: ${new Date().toISOString()}</div>

        <div class="summary">
            <div class="summary-card">
                <div class="summary-value">${blocks.length}</div>
                <div class="summary-label">Total Blocks</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">${provenance.length}</div>
                <div class="summary-label">Provenance Entries</div>
            </div>
            <div class="summary-card">
                <div class="summary-value">${formatSize(fileInfo.size || 0)}</div>
                <div class="summary-label">File Size</div>
            </div>
        </div>

        <h2>File Information</h2>
        <div class="section">
            <div class="stat-row"><span>Format</span><span>${parser.format || 'unknown'}</span></div>
            <div class="stat-row"><span>Version</span><span>${fileInfo.version || 'N/A'}</span></div>
            <div class="stat-row"><span>Created By</span><span>${fileInfo.agent || 'N/A'}</span></div>
        </div>

        <h2>Block Statistics</h2>
        <div class="section">${blockStatsHtml}</div>

        <h2>Security</h2>
        <div class="section">
            <div class="stat-row"><span>Header Signature</span><span>${parser.securityInfo?.signature ? 'Present' : 'None'}</span></div>
            <div class="stat-row"><span>Signed Blocks</span><span>${blocks.filter(b => b.signature).length}/${blocks.length}</span></div>
            <div class="stat-row"><span>Merkle Root</span><span>${parser.securityInfo?.merkleRoot ? 'Present' : 'None'}</span></div>
        </div>

        <h2>Provenance Summary</h2>
        <div class="section">
            <div class="stat-row"><span>Genesis</span><span>${provenance.find(p => p.action === 'genesis') ? 'Yes' : 'No'}</span></div>
            <div class="stat-row"><span>Total Operations</span><span>${provenance.length}</span></div>
            <div class="stat-row"><span>Agents</span><span>${new Set(provenance.map(p => p.agent_id)).size}</span></div>
        </div>
    </body>
    </html>`;
}

export function deactivate() {}

