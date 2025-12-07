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
    ['link', { rel: 'icon', href: '/maif/favicon.ico' }],
    ['link', { rel: 'mask-icon', href: '/maif/safari-pinned-tab.svg', color: '#3c82f6' }],
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
  base: '/maif/',
  
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
    base: '/maif/',
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
    logo: '/maif/maif-logo.svg',
    siteTitle: 'MAIF Framework',
    
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Integrations', link: '/guide/integrations/' },
      { text: 'Examples', link: '/examples/' },
      { 
        text: 'API Reference', 
        link: 'https://deepwiki.com/vineethsai/maif/3-api-reference',
        target: '_blank'
      },
      { 
        text: 'DeepWiki', 
        link: 'https://deepwiki.com/vineethsai/maif',
        target: '_blank'
      },
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
            { text: 'Quick Start', link: '/guide/getting-started' },
            { text: 'Installation', link: '/guide/installation' }
          ]
        },
        {
          text: 'Framework Integrations',
          collapsed: false,
          items: [
            { text: 'Overview', link: '/guide/integrations/' },
            { text: 'LangGraph', link: '/guide/integrations/langgraph' },
            { text: 'LangChain', link: '/guide/integrations/langchain' },
            { text: 'CrewAI', link: '/guide/integrations/crewai' },
            { text: 'Strands SDK', link: '/guide/integrations/strands' }
          ]
        },
        {
          text: 'Deep Dive (DeepWiki)',
          items: [
            { text: 'Core Concepts', link: 'https://deepwiki.com/vineethsai/maif/2-core-concepts' },
            { text: 'API Reference', link: 'https://deepwiki.com/vineethsai/maif/3-api-reference' },
            { text: 'Security Model', link: 'https://deepwiki.com/vineethsai/maif/2.2-cryptographic-security' },
            { text: 'Development Guide', link: 'https://deepwiki.com/vineethsai/maif/6-development-guide' }
          ]
        },
        {
          text: 'Legacy Pages',
          collapsed: true,
          items: [
            { text: 'Concepts (deprecated)', link: '/guide/concepts' },
            { text: 'Architecture (deprecated)', link: '/guide/architecture' },
            { text: 'Blocks (deprecated)', link: '/guide/blocks' },
            { text: 'Privacy (deprecated)', link: '/guide/privacy' },
            { text: 'Security Model (deprecated)', link: '/guide/security-model' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview (redirects to DeepWiki)', link: '/api/' }
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Featured Examples',
          items: [
            { text: 'Overview', link: '/examples/' },
            { text: 'LangGraph Multi-Agent RAG', link: '/examples/langgraph-rag' },
            { text: 'CrewAI Research Crew', link: '/examples/crewai-research' }
          ]
        },
        {
          text: 'Quick Start',
          items: [
            { text: 'Hello World', link: '/examples/hello-world' }
          ]
        },
        {
          text: 'More Examples (DeepWiki)',
          items: [
            { text: 'All Examples', link: 'https://deepwiki.com/vineethsai/maif/5-examples-and-use-cases' },
            { text: 'Multi-Agent Systems', link: 'https://deepwiki.com/vineethsai/maif/5.3-multi-agent-systems' },
            { text: 'AWS Integration', link: 'https://deepwiki.com/vineethsai/maif/5.4-aws-integration-examples' }
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