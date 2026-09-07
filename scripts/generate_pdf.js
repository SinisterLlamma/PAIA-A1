const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Path configurations
const projectDir = path.resolve(__dirname, '..');
const reportMd = path.join(projectDir, 'report', 'preliminary_report.md');
const reportHtml = path.join(projectDir, 'report', 'preliminary_report_preview.html');
const reportPdf = path.join(projectDir, 'report', 'preliminary_report.pdf');
const playwrightPkg = '/home/harsha/.cache/ms-playwright-go/1.50.1/package';
const chromeExecutable = '/home/harsha/.cache/ms-playwright/chromium-1155/chrome-linux/chrome';

console.log('>>> Converting Markdown to HTML via Pandoc...');
const pandocCmd = `pandoc "${reportMd}" --from=markdown+tex_math_dollars+pipe_tables --to=html5 --mathjax`;
const htmlBody = execSync(pandocCmd, { encoding: 'utf-8' });

console.log('>>> Building styled standalone HTML document with GitHub Markdown & MathJax...');
const fullHtml = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CUDA SGEMM Optimization & Architectural Analysis Report</title>
  <!-- GitHub Markdown CSS -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.5.1/github-markdown.min.css">
  <!-- Highlight.js for code blocks -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <!-- MathJax for TeX vector math -->
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true
      },
      svg: {
        fontCache: 'global'
      }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

  <style>
    @page {
      size: A4;
      margin: 16mm 14mm 16mm 14mm;
    }
    body {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      background-color: #ffffff;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
      color: #24292f;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
    .markdown-body {
      max-width: 100% !important;
      padding: 0 4px !important;
      font-size: 13.5px !important;
      line-height: 1.65 !important;
      background-color: transparent !important;
    }
    .markdown-body h1 {
      font-size: 24px !important;
      border-bottom: 2px solid #0969da !important;
      padding-bottom: 8px !important;
      margin-top: 10px !important;
      color: #1f2328 !important;
    }
    .markdown-body h2 {
      font-size: 18px !important;
      border-bottom: 1px solid #d0d7de !important;
      padding-bottom: 6px !important;
      margin-top: 24px !important;
      page-break-after: avoid;
      break-after: avoid;
      color: #0969da !important;
    }
    .markdown-body h3 {
      font-size: 15px !important;
      margin-top: 18px !important;
      page-break-after: avoid;
      break-after: avoid;
      color: #24292f !important;
    }
    .markdown-body h4 {
      font-size: 14px !important;
      margin-top: 14px !important;
      page-break-after: avoid;
      break-after: avoid;
    }
    .markdown-body table {
      width: 100% !important;
      display: table !important;
      border-collapse: collapse !important;
      margin: 14px 0 !important;
      font-size: 11.5px !important;
      page-break-inside: auto;
      break-inside: auto;
    }
    .markdown-body table th {
      background-color: #f6f8fa !important;
      font-weight: 600 !important;
      padding: 6px 10px !important;
      border: 1px solid #d0d7de !important;
      text-align: left;
    }
    .markdown-body table td {
      padding: 6px 10px !important;
      border: 1px solid #d0d7de !important;
    }
    .markdown-body table tr:nth-child(2n) {
      background-color: #fbfcfd !important;
    }
    .markdown-body table tr {
      page-break-inside: avoid;
      break-inside: avoid;
    }
    .markdown-body img {
      max-width: 90% !important;
      height: auto !important;
      display: block !important;
      margin: 16px auto !important;
      border-radius: 6px !important;
      border: 1px solid #d0d7de !important;
      box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06) !important;
      page-break-inside: avoid;
      break-inside: avoid;
    }
    .markdown-body pre {
      background-color: #f6f8fa !important;
      border: 1px solid #d0d7de !important;
      border-radius: 6px !important;
      padding: 12px !important;
      font-size: 11.5px !important;
      page-break-inside: avoid;
      break-inside: avoid;
      overflow-x: hidden !important;
      white-space: pre-wrap !important;
    }
    .markdown-body code {
      font-size: 88% !important;
      background-color: rgba(175, 184, 193, 0.2) !important;
      border-radius: 4px !important;
      padding: 0.15em 0.35em !important;
      font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace !important;
    }
    .markdown-body pre code {
      background-color: transparent !important;
      padding: 0 !important;
    }
    .markdown-body blockquote {
      border-left: 4px solid #0969da !important;
      background-color: #f6f8fa !important;
      padding: 8px 16px !important;
      margin: 14px 0 !important;
      color: #57606a !important;
      border-radius: 0 4px 4px 0;
      page-break-inside: avoid;
      break-inside: avoid;
    }
    .markdown-body hr {
      border: 0;
      height: 1px;
      background: #d0d7de;
      margin: 20px 0;
    }
    mjx-container {
      margin: 0 !important;
      padding: 0 !important;
      display: inline-block !important;
    }
    mjx-container[jax="SVG"][display="true"] {
      display: block !important;
      margin: 10px 0 !important;
      text-align: center !important;
    }
  </style>
</head>
<body>
  <article class="markdown-body">
    ${htmlBody}
  </article>
  <script>
    hljs.highlightAll();
  </script>
</body>
</html>`;

fs.writeFileSync(reportHtml, fullHtml, 'utf-8');
console.log(`>>> Styled HTML preview generated: ${reportHtml}`);

console.log('>>> Launching Chromium via Playwright to generate PDF...');
const { chromium } = require(playwrightPkg);

(async () => {
  const browser = await chromium.launch({
    executablePath: chromeExecutable,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu']
  });

  const page = await browser.newPage();
  
  // Navigate to local HTML file
  await page.goto(`file://${reportHtml}`, { waitUntil: 'networkidle' });

  // Wait for MathJax to finish typesetting
  console.log('>>> Waiting for MathJax rendering...');
  await page.waitForFunction(() => {
    return window.MathJax && window.MathJax.startup && window.MathJax.startup.promise
      ? window.MathJax.startup.promise.then(() => true)
      : true;
  }, { timeout: 30000 });

  // Wait for all images to be loaded and decoded
  console.log('>>> Waiting for images to load...');
  await page.evaluate(async () => {
    const images = Array.from(document.images);
    await Promise.all(images.map(img => {
      if (img.complete) return Promise.resolve();
      return new Promise(resolve => {
        img.onload = resolve;
        img.onerror = resolve;
      });
    }));
  });

  // Brief stabilization delay for layout settling
  await new Promise(r => setTimeout(r, 1500));

  console.log('>>> Printing to PDF with exact A4 markdown margins and pagination...');
  await page.pdf({
    path: reportPdf,
    format: 'A4',
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<div style="font-size: 8px; color: #8c959f; width: 100%; text-align: right; padding-right: 14mm; font-family: -apple-system, sans-serif;">CUDA SGEMM Microarchitectural Performance Report — NVIDIA RTX 3090</div>',
    footerTemplate: '<div style="font-size: 8px; color: #8c959f; width: 100%; text-align: center; font-family: -apple-system, sans-serif;">Page <span class="pageNumber"></span> of <span class="totalPages"></span></div>',
    margin: {
      top: '18mm',
      bottom: '18mm',
      left: '14mm',
      right: '14mm'
    }
  });

  await browser.close();
  const stats = fs.statSync(reportPdf);
  console.log(`=======================================================`);
  console.log(`SUCCESS: PDF Report Generated!`);
  console.log(`Path: ${reportPdf}`);
  console.log(`Size: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
  console.log(`=======================================================`);
})();
