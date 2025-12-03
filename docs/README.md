# MAIF Documentation

This is the official documentation for MAIF (Multi-Agent Intelligence Framework) - a revolutionary memory framework for AI agents with built-in privacy, semantic understanding, and enterprise-grade security.

## 🚀 Live Documentation

- **Production**: [https://maif.ai/docs](https://maif.ai/docs)
- **Staging**: [https://staging-docs.maif.ai](https://staging-docs.maif.ai)

## 📁 Documentation Structure

```
docs/
├── .vitepress/          # VitePress configuration
│   ├── config.js        # Site configuration
│   └── theme/           # Custom theme files
├── guide/               # User guides and tutorials
│   ├── getting-started.md
│   ├── installation.md
│   ├── quick-start.md
│   └── concepts.md
├── api/                 # API reference documentation
│   ├── core/            # Core API documentation
│   ├── privacy/         # Privacy & security APIs
│   ├── semantic/        # Semantic processing APIs
│   └── streaming/       # Streaming & performance APIs
├── examples/            # Real-world examples
│   ├── financial-agent.md
│   ├── healthcare-agent.md
│   └── content-moderation.md
├── cookbook/            # Advanced patterns and recipes
│   ├── performance.md
│   ├── security.md
│   └── deployment.md
└── public/              # Static assets
    ├── images/
    ├── favicon.ico
    └── maif-logo.svg
```

## 🛠️ Local Development

### Prerequisites

- **Node.js**: 18.0.0 or higher
- **npm**: 9.0.0 or higher
- **Git**: Latest version

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/maif-ai/maif.git
cd maif/docs

# Install dependencies
npm install

# Start development server
npm run dev

# Open in browser
open http://localhost:5173
```

### Available Scripts

```bash
# Development
npm run dev              # Start development server with hot reload
npm run build            # Build for production
npm run preview          # Preview production build locally

# Quality Assurance  
npm run lint             # Lint all files
npm run lint:fix         # Fix linting issues automatically
npm run type-check       # TypeScript type checking
npm run test             # Run tests
npm run test:ui          # Run tests with UI

# Deployment
npm run build:ci         # CI/CD build (includes linting and type checking)
npm run deploy           # Deploy to GitHub Pages
npm run sitemap          # Generate sitemap
```

## 🏗️ Building Documentation

### Production Build

```bash
# Full production build
npm run build:ci

# Outputs to .vitepress/dist/
# Ready for deployment to any static hosting service
```

### Performance Optimization

The documentation is optimized for:

- **Fast Loading**: <2s initial page load
- **SEO**: Comprehensive meta tags and structured data
- **Accessibility**: WCAG 2.1 AA compliance
- **Mobile**: Responsive design for all devices
- **Search**: Built-in search with MiniSearch

### Bundle Analysis

```bash
# Analyze bundle size
npm run build
npx vite-bundle-analyzer .vitepress/dist
```

## 🚀 Deployment

### GitHub Pages (Recommended)

```bash
# Deploy to GitHub Pages
npm run deploy

# Custom domain setup (if needed)
echo "docs.maif.ai" > .vitepress/dist/CNAME
```

### Netlify

```bash
# Build command
npm run build

# Publish directory  
.vitepress/dist

# Environment variables
NODE_VERSION=18
```

### Vercel

```bash
# Build command
npm run build

# Output directory
.vitepress/dist

# Framework preset
VitePress
```

### Docker Deployment

```dockerfile
# docs/Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy documentation source
COPY . .

# Build documentation
RUN npm run build

# Serve with nginx
FROM nginx:alpine
COPY --from=0 /app/.vitepress/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

```bash
# Build and run
docker build -t maif-docs .
docker run -p 8080:80 maif-docs
```

### Kubernetes Deployment

```yaml
# docs/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: maif-docs
spec:
  replicas: 3
  selector:
    matchLabels:
      app: maif-docs
  template:
    metadata:
      labels:
        app: maif-docs
    spec:
      containers:
      - name: docs
        image: maif-docs:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: maif-docs-service
spec:
  selector:
    app: maif-docs
  ports:
  - port: 80
    targetPort: 80
  type: LoadBalancer
```

## 📝 Contributing to Documentation

### Writing Guidelines

1. **Clarity**: Write clear, concise explanations
2. **Examples**: Include runnable code examples
3. **Structure**: Use consistent heading structure
4. **Links**: Link to related sections and external resources
5. **Testing**: Test all code examples before submitting

### Content Standards

- **Code Examples**: Must be fully runnable
- **Performance**: Include performance considerations
- **Security**: Highlight security best practices
- **Accessibility**: Follow accessibility guidelines
- **SEO**: Optimize for search engines

### Adding New Pages

```bash
# Create new page
touch guide/my-new-guide.md

# Add to navigation (.vitepress/config.js)
sidebar: {
  '/guide/': [
    // ... existing items
    { text: 'My New Guide', link: '/guide/my-new-guide' }
  ]
}
```

### Content Style Guide

```markdown
# Page Title (H1)

Brief introduction paragraph explaining what this page covers.

## Section Title (H2)

### Subsection (H3)

```python
# Code examples should be complete and runnable
from maif_api import create_maif

maif = create_maif("example-agent")
maif.add_text("Hello!")
print("✅ Example works!")
```

::: tip Pro Tip
Use callout boxes for important information.
:::

::: warning Important
Use warnings for critical information.
:::

::: danger Security
Use danger boxes for security-related warnings.
:::
```

## 🔧 Advanced Configuration

### Custom Theme

```javascript
// .vitepress/theme/index.js
import DefaultTheme from 'vitepress/theme'
import './custom.css'
import CustomComponent from './components/CustomComponent.vue'

export default {
  ...DefaultTheme,
  enhanceApp({ app }) {
    app.component('CustomComponent', CustomComponent)
  }
}
```

### Search Configuration

```javascript
// .vitepress/config.js
export default {
  themeConfig: {
    search: {
      provider: 'local',
      options: {
        miniSearch: {
          searchOptions: {
            fuzzy: 0.2,
            prefix: true,
            boost: { title: 4, text: 2, titles: 1 }
          }
        }
      }
    }
  }
}
```

### Analytics Setup

```javascript
// .vitepress/config.js
export default {
  head: [
    // Google Analytics
    ['script', { async: '', src: 'https://www.googletagmanager.com/gtag/js?id=G-MAIF-ANALYTICS' }],
    ['script', {}, `
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-MAIF-ANALYTICS');
    `]
  ]
}
```

## 🧪 Testing

### Link Checking

```bash
# Install link checker
npm install -g markdown-link-check

# Check all markdown files
find . -name "*.md" -exec markdown-link-check {} \;
```

### Accessibility Testing

```bash
# Install accessibility checker
npm install -g @axe-core/cli

# Test built site
axe http://localhost:5173
```

### Performance Testing

```bash
# Install Lighthouse CI
npm install -g @lhci/cli

# Run Lighthouse tests
lhci autorun
```

## 📊 Monitoring & Analytics

### Performance Metrics

- **Core Web Vitals**: LCP, FID, CLS
- **Loading Speed**: Time to first byte, first contentful paint
- **Bundle Size**: JavaScript and CSS bundle sizes
- **Search Performance**: Search query response times

### Content Analytics

- **Page Views**: Most popular documentation pages
- **Search Queries**: What users search for
- **User Flow**: How users navigate through docs
- **Bounce Rate**: Pages where users leave quickly

### Error Monitoring

```javascript
// Error tracking setup
window.addEventListener('error', (event) => {
  // Send to monitoring service (e.g., Sentry)
  console.error('Documentation error:', event.error)
})
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/docs.yml
name: Deploy Documentation

on:
  push:
    branches: [main]
    paths: ['docs/**']
  pull_request:
    paths: ['docs/**']

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
        cache-dependency-path: docs/package-lock.json
    
    - name: Install dependencies
      run: |
        cd docs
        npm ci
    
    - name: Lint and type check
      run: |
        cd docs
        npm run lint
        npm run type-check
    
    - name: Build documentation
      run: |
        cd docs
        npm run build
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: docs/.vitepress/dist
        cname: docs.maif.ai
```

### Quality Gates

- ✅ **Linting**: ESLint passes
- ✅ **Type Checking**: TypeScript compilation successful
- ✅ **Link Checking**: All internal links valid
- ✅ **Accessibility**: Meets WCAG 2.1 AA
- ✅ **Performance**: Lighthouse score >90
- ✅ **Bundle Size**: <500KB gzipped

## 🆘 Troubleshooting

### Common Issues

#### Build Failures

```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear VitePress cache
rm -rf .vitepress/cache
npm run build
```

#### Development Server Issues

```bash
# Check port availability
lsof -ti:5173

# Use different port
npm run dev -- --port 3000
```

#### Search Not Working

```bash
# Rebuild search index
rm -rf .vitepress/cache
npm run build
```

### Performance Issues

```bash
# Bundle analysis
npm run build
npx vite-bundle-analyzer .vitepress/dist

# Optimize images
npm install -g imagemin-cli
imagemin public/images/* --out-dir=public/images/optimized
```

## 📞 Support

- **Documentation Issues**: [GitHub Issues](https://github.com/maif-ai/maif/issues)
- **General Support**: [Discord Community](https://discord.gg/maif)
- **Feature Requests**: [GitHub Discussions](https://github.com/maif-ai/maif/discussions)

## 📄 License

This documentation is licensed under [MIT License](../LICENSE).

---

**Built with ❤️ using [VitePress](https://vitepress.dev/)** 