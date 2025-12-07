/**
 * VSCode MAIF Extension Tests
 * 
 * Run with: npm test
 */

import * as assert from 'assert';
import * as vscode from 'vscode';

// Mock MAIFParser for testing
class MockMAIFParser {
    manifest: any = null;
    blocks: any[] = [];
    provenance: any[] = [];
    fileInfo: any = null;
    securityInfo: any = null;
    format: string = 'secure';
    binaryData: Uint8Array | null = null;

    loadManifestFromString(text: string): void {
        this.manifest = JSON.parse(text);
    }

    loadBinaryFromBuffer(buffer: Uint8Array): void {
        this.binaryData = buffer;
        // Parse header
        if (buffer.length >= 4) {
            const magic = String.fromCharCode(...buffer.slice(0, 4));
            if (magic === 'MAIF') {
                this.format = 'secure';
            }
        }
    }

    static formatTimestamp(ts: number): string {
        const ms = ts > 1e12 ? ts / 1000 : ts * 1000;
        return new Date(ms).toLocaleString();
    }

    static formatSize(bytes: number): string {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    static BLOCK_TYPES: Record<string, { name: string; icon: string; color: string }> = {
        'TEXT': { name: 'Text', icon: 'T', color: '#3b82f6' },
        'EMBD': { name: 'Embeddings', icon: 'E', color: '#8b5cf6' },
    };
}

suite('MAIF Extension Test Suite', () => {
    vscode.window.showInformationMessage('Starting MAIF Extension tests');

    // Parser Tests
    suite('MAIFParser', () => {
        test('should parse manifest JSON', () => {
            const parser = new MockMAIFParser();
            const manifest = {
                maif_version: '3.0',
                agent_id: 'test-agent',
                blocks: [],
            };
            parser.loadManifestFromString(JSON.stringify(manifest));
            
            assert.strictEqual(parser.manifest.maif_version, '3.0');
            assert.strictEqual(parser.manifest.agent_id, 'test-agent');
        });

        test('should detect secure format from magic bytes', () => {
            const parser = new MockMAIFParser();
            const buffer = new Uint8Array([0x4D, 0x41, 0x49, 0x46, 0x00, 0x03]); // "MAIF" + version
            parser.loadBinaryFromBuffer(buffer);
            
            assert.strictEqual(parser.format, 'secure');
        });

        test('should format timestamps correctly', () => {
            // Unix seconds
            const ts1 = MockMAIFParser.formatTimestamp(1700000000);
            assert.ok(ts1.includes('2023'));
            
            // Microseconds
            const ts2 = MockMAIFParser.formatTimestamp(1700000000000000);
            assert.ok(ts2.includes('2023'));
        });

        test('should format file sizes correctly', () => {
            assert.strictEqual(MockMAIFParser.formatSize(0), '0 B');
            assert.strictEqual(MockMAIFParser.formatSize(1024), '1 KB');
            assert.strictEqual(MockMAIFParser.formatSize(1048576), '1 MB');
            assert.strictEqual(MockMAIFParser.formatSize(1536), '1.5 KB');
        });
    });

    // Integrity Verification Tests
    suite('Integrity Verification', () => {
        interface IntegrityResult {
            isValid: boolean;
            errors: string[];
            warnings: string[];
            checks: { name: string; passed: boolean; details: string }[];
        }

        function verifyArtifactIntegrity(parser: MockMAIFParser): IntegrityResult {
            const result: IntegrityResult = {
                isValid: true,
                errors: [],
                warnings: [],
                checks: []
            };

            // Check header signature
            result.checks.push({
                name: 'Header Signature',
                passed: !!parser.securityInfo?.signature,
                details: parser.securityInfo?.signature ? 'Ed25519 signature present' : 'No signature found'
            });

            // Check blocks
            let chainValid = true;
            for (const block of parser.blocks || []) {
                if (block.isTampered) {
                    chainValid = false;
                    result.errors.push(`Block at offset ${block.offset} is tampered`);
                }
            }
            result.checks.push({
                name: 'Block Chain',
                passed: chainValid,
                details: chainValid ? 'Chain verified' : 'Chain compromised'
            });

            result.isValid = result.errors.length === 0;
            return result;
        }

        test('should detect valid artifact', () => {
            const parser = new MockMAIFParser();
            parser.securityInfo = { signature: 'abc123' };
            parser.blocks = [
                { type: 'TEXT', offset: 0, isTampered: false },
                { type: 'TEXT', offset: 100, isTampered: false },
            ];
            
            const result = verifyArtifactIntegrity(parser);
            assert.strictEqual(result.isValid, true);
            assert.strictEqual(result.errors.length, 0);
        });

        test('should detect tampered blocks', () => {
            const parser = new MockMAIFParser();
            parser.blocks = [
                { type: 'TEXT', offset: 0, isTampered: false },
                { type: 'TEXT', offset: 100, isTampered: true },
            ];
            
            const result = verifyArtifactIntegrity(parser);
            assert.strictEqual(result.isValid, false);
            assert.ok(result.errors.length > 0);
        });

        test('should report missing signature', () => {
            const parser = new MockMAIFParser();
            parser.securityInfo = null;
            
            const result = verifyArtifactIntegrity(parser);
            const sigCheck = result.checks.find(c => c.name === 'Header Signature');
            assert.strictEqual(sigCheck?.passed, false);
        });
    });

    // Session Extraction Tests
    suite('Session Extraction', () => {
        interface Session {
            id: string;
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
                        blockCount: 0,
                        type: meta.type || 'unknown'
                    });
                }

                const session = sessions.get(threadId)!;
                session.blockCount++;
            }

            return Array.from(sessions.values());
        }

        test('should extract sessions from blocks', () => {
            const blocks = [
                { metadata: { thread_id: 'session-1', type: 'checkpoint' } },
                { metadata: { thread_id: 'session-1', type: 'checkpoint' } },
                { metadata: { thread_id: 'session-2', type: 'checkpoint' } },
            ];
            
            const sessions = extractSessions(blocks);
            assert.strictEqual(sessions.length, 2);
            
            const session1 = sessions.find(s => s.id === 'session-1');
            assert.strictEqual(session1?.blockCount, 2);
        });

        test('should handle blocks without session', () => {
            const blocks = [
                { metadata: {} },
                { metadata: {} },
            ];
            
            const sessions = extractSessions(blocks);
            assert.strictEqual(sessions.length, 1);
            assert.strictEqual(sessions[0].id, 'default');
            assert.strictEqual(sessions[0].blockCount, 2);
        });

        test('should handle empty blocks array', () => {
            const sessions = extractSessions([]);
            assert.strictEqual(sessions.length, 0);
        });
    });

    // Block Search Tests
    suite('Block Search', () => {
        function searchBlocks(blocks: any[], query: string): any[] {
            const queryLower = query.toLowerCase();
            return blocks.filter(block => {
                if (block.type?.toLowerCase().includes(queryLower)) return true;
                const metaStr = JSON.stringify(block.metadata || {}).toLowerCase();
                if (metaStr.includes(queryLower)) return true;
                if (block.content?.toLowerCase().includes(queryLower)) return true;
                return false;
            });
        }

        test('should find blocks by type', () => {
            const blocks = [
                { type: 'TEXT', metadata: {} },
                { type: 'EMBD', metadata: {} },
                { type: 'TEXT', metadata: {} },
            ];
            
            const results = searchBlocks(blocks, 'TEXT');
            assert.strictEqual(results.length, 2);
        });

        test('should find blocks by metadata', () => {
            const blocks = [
                { type: 'TEXT', metadata: { thread_id: 'user-alice' } },
                { type: 'TEXT', metadata: { thread_id: 'user-bob' } },
            ];
            
            const results = searchBlocks(blocks, 'alice');
            assert.strictEqual(results.length, 1);
        });

        test('should find blocks by content', () => {
            const blocks = [
                { type: 'TEXT', content: 'Hello world' },
                { type: 'TEXT', content: 'Goodbye world' },
            ];
            
            const results = searchBlocks(blocks, 'hello');
            assert.strictEqual(results.length, 1);
        });

        test('should be case insensitive', () => {
            const blocks = [
                { type: 'TEXT', content: 'HELLO WORLD' },
            ];
            
            const results = searchBlocks(blocks, 'hello');
            assert.strictEqual(results.length, 1);
        });
    });
});

