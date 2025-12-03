import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: 'MAIF Framework',
  description: 'Multi-Agent Intelligence Framework - Cutting-edge memory framework for AI agent systems with advanced privacy, semantic understanding, and high-performance capabilities',
  
  // Ignore dead links for now to allow deployment
  ignoreDeadLinks: true,
  
  head: [
    ['meta', { name: 'theme-color', content: '#3c82f6' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'en' }],
    ['meta', { name: 'og:site_name', content: 'MAIF Framework' }],
    ['meta', { name: 'og:image', content: '/maif-og-image.png' }],
    ['link', { rel: 'icon', href: '/maifscratch-1/favicon.ico' }],
    ['link', { rel: 'mask-icon', href: '/maifscratch-1/safari-pinned-tab.svg', color: '#3c82f6' }],
    ['meta', { name: 'msapplication-TileColor', content: '#3c82f6' }],
    // Custom CSS for code overflow and styling
    ['style', {}, `
      .vp-code-group .tabs { overflow-x: auto; }
      .vp-code-group .tabs::-webkit-scrollbar { height: 4px; }
      .vp-code-group .tabs::-webkit-scrollbar-thumb { background: var(--vp-c-divider); border-radius: 2px; }
      .language-python, .language-javascript, .language-typescript, .language-bash { 
        overflow-x: auto; 
        word-wrap: break-word; 
      }
      pre code { 
        white-space: pre; 
        overflow-x: auto; 
        display: block; 
        padding: 1rem; 
      }
      .mermaid-rendered { 
        text-align: center; 
        margin: 2rem 0; 
        overflow-x: auto; 
      }
      .mermaid-error {
        color: #ef4444;
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
      }
      @media (max-width: 768px) {
        pre code { font-size: 12px; }
        .mermaid-rendered { font-size: 12px; }
      }
    `]
  ],

  cleanUrls: true,
  lastUpdated: true,
  
  // Set base URL for GitHub Pages deployment
  base: '/maifscratch-1/',
  
  // Ensure proper asset handling for GitHub Pages
  outDir: '.vitepress/dist',
  
  // Markdown configuration - remove any conflicting mermaid config
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true,
    // Let our custom theme handle mermaid rendering
    config: (md) => {
      // Custom markdown configurations can go here
    }
  },

  // Simplified Vite configuration for GitHub Pages
  vite: {
    base: '/maifscratch-1/',
    optimizeDeps: {
      include: ['mermaid']
    },
    ssr: {
      noExternal: ['mermaid']
    }
  },

  // Mermaid configuration
  mermaid: {
    theme: 'default',
    flowchart: {
      useMaxWidth: false,
      htmlLabels: true,
      nodeSpacing: 50,
      rankSpacing: 50
    },
    sequence: {
      useMaxWidth: false,
      diagramMarginX: 50,
      diagramMarginY: 50,
      actorMargin: 50,
      width: 150,
      height: 65,
      boxMargin: 10,
      boxTextMargin: 5,
      noteMargin: 10,
      messageMargin: 35
    },
    gantt: {
      useMaxWidth: false
    },
    journey: {
      useMaxWidth: false
    },
    class: {
      useMaxWidth: false
    },
    state: {
      useMaxWidth: false
    },
    er: {
      useMaxWidth: false
    }
  },
  
  themeConfig: {
    logo: '/maifscratch-1/maif-logo.svg',
    siteTitle: 'MAIF Framework',
    
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API Reference', link: '/api/' },
      { text: 'Examples', link: '/examples/' },
      {
        text: 'v1.0.0',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'Contributing', link: '/contributing' }
        ]
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quick Start', link: '/guide/quick-start' },
            { text: 'Getting Started', link: '/guide/getting-started' }
          ]
        },
        {
          text: 'Core Concepts',
          items: [
            { text: 'Overview', link: '/guide/concepts' },
            { text: 'Architecture', link: '/guide/architecture' },
            { text: 'Blocks & Artifacts', link: '/guide/blocks' }
          ]
        },
        {
          text: 'Features',
          items: [
            { text: 'Performance', link: '/guide/performance' },
            { text: 'Streaming', link: '/guide/streaming' },
            { text: 'Semantic Understanding', link: '/guide/semantic' },
            { text: 'Multi-modal Data', link: '/guide/multimodal' },
            { text: 'ACID Transactions', link: '/guide/acid' }
          ]
        },
        {
          text: 'Privacy & Security',
          items: [
            { text: 'Privacy Framework', link: '/guide/privacy' },
            { text: 'Security Model', link: '/guide/security-model' }
          ]
        },
        {
          text: 'Advanced Topics',
          items: [
            { text: 'Agent Development', link: '/guide/agent-development' },
            { text: 'Agent Lifecycle', link: '/guide/agent-lifecycle' },
            { text: 'Distributed Deployment', link: '/guide/distributed' },
            { text: 'Monitoring', link: '/guide/monitoring' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview', link: '/api/' }
          ]
        },
        {
          text: 'Core API',
          items: [
            { text: 'MAIFClient', link: '/api/core/client' },
            { text: 'Artifact', link: '/api/core/artifact' },
            { text: 'Encoder/Decoder', link: '/api/core/encoder-decoder' }
          ]
        },
        {
          text: 'Privacy API',
          items: [
            { text: 'Privacy Engine', link: '/api/privacy/engine' }
          ]
        },
        {
          text: 'Security API',
          items: [
            { text: 'Overview', link: '/api/security/' },
            { text: 'Access Control', link: '/api/security/access-control' },
            { text: 'Cryptography', link: '/api/security/crypto' }
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Examples',
          items: [
            { text: 'Overview', link: '/examples/' },
            { text: 'Financial Agent', link: '/examples/financial-agent' }
          ]
        },
        {
          text: 'Real-world Use Cases',
          items: [
            { text: 'Healthcare AI Agent', link: '/examples/healthcare-agent' },
            { text: 'Content Moderation', link: '/examples/content-moderation' },
            { text: 'Research Assistant', link: '/examples/research-assistant' },
            { text: 'Security Monitor', link: '/examples/security-monitor' }
          ]
        },
        {
          text: 'Integration Examples',
          items: [
            { text: 'LangChain Integration', link: '/examples/langchain' },
            { text: 'Hugging Face Models', link: '/examples/huggingface' },
            { text: 'Ray/Dask Distributed', link: '/examples/distributed' },
            { text: 'Kafka Streaming', link: '/examples/kafka' }
          ]
        }
      ]
    },

    editLink: {
      pattern: 'https://github.com/maif-ai/maif/edit/main/docs/:path'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/maif-ai/maif' },
      { icon: 'discord', link: 'https://discord.gg/maif' },
      { icon: 'twitter', link: 'https://twitter.com/maif_ai' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 MAIF Contributors'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: 'Search MAIF docs'
          }
        }
      }
    }
  },

  vue: {
    reactivityTransform: true
  },

  // Add client-side Mermaid initialization
  buildEnd() {
    // This will be handled by the theme
  },

  // PWA configuration
  pwa: {
    outDir: '.vitepress/dist',
    registerType: 'autoUpdate',
    includeAssets: ['favicon.ico', 'apple-touch-icon.png'],
    manifest: {
      name: 'MAIF Framework Documentation',
      short_name: 'MAIF Docs',
      description: 'Multi-Agent Intelligence Framework Documentation',
      theme_color: '#3c82f6',
      icons: [
        {
          src: 'pwa-192x192.png',
          sizes: '192x192',
          type: 'image/png'
        },
        {
          src: 'pwa-512x512.png',
          sizes: '512x512',
          type: 'image/png'
        }
      ]
    }
  }
})) 