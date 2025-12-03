import DefaultTheme from 'vitepress/theme'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()
    
    const initializeZoom = async () => {
      if (typeof window === 'undefined') return
      
      try {
        // Dynamic import of medium-zoom
        const mediumZoom = (await import('medium-zoom')).default
        
        // Initialize zoom for mermaid diagrams
        mediumZoom('.mermaid svg, .mermaid-diagram svg, [data-type="mermaid"] svg', {
          background: 'rgba(0, 0, 0, 0.8)',
          scrollOffset: 40,
          margin: 40
        })
        
        // Also zoom regular images
        mediumZoom('img:not(.no-zoom)', {
          background: 'rgba(0, 0, 0, 0.8)',
          scrollOffset: 40,
          margin: 40
        })
      } catch (error) {
        console.error('Failed to initialize zoom:', error)
      }
    }

    const enhanceMermaidDiagrams = () => {
      // Find all mermaid containers and make them more responsive
      const mermaidContainers = document.querySelectorAll('.mermaid, .mermaid-diagram, [data-type="mermaid"]')
      
      mermaidContainers.forEach(container => {
        const svg = container.querySelector('svg')
        if (svg) {
          // Make SVG responsive
          svg.style.maxWidth = '100%'
          svg.style.height = 'auto'
          svg.style.cursor = 'zoom-in'
          
          // Add a wrapper for better styling
          if (!container.classList.contains('mermaid-enhanced')) {
            container.classList.add('mermaid-enhanced')
            container.style.textAlign = 'center'
            container.style.margin = '2rem 0'
            container.style.padding = '1rem'
            container.style.border = '1px solid var(--vp-c-divider)'
            container.style.borderRadius = '8px'
            container.style.backgroundColor = 'var(--vp-c-bg-soft)'
            container.style.overflow = 'auto'
          }
        }
      })
    }

    const initializePage = () => {
      nextTick(() => {
        setTimeout(() => {
          enhanceMermaidDiagrams()
          initializeZoom()
        }, 500) // Give mermaid time to render
      })
    }

    onMounted(() => {
      initializePage()
    })
    
    // Re-render on route change
    watch(
      () => route.path,
      () => {
        initializePage()
      }
    )
  }
} 