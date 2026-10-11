/**
 * CoverLetterEditor – Rich text editor for generated cover letters.
 * Supports formatting controls and download as a Word-compatible .doc file.
 */

import { useState, useRef } from 'react';
import {
  Copy,
  Download,
  CheckCheck,
  FileText,
  Sparkles,
  Bold,
  Italic,
  Underline,
  List,
  ListOrdered,
  AlignLeft,
  AlignCenter,
  AlignRight,
} from 'lucide-react';

export default function CoverLetterEditor({ coverLetterText, jobTitle, company }) {
  const escapeHtml = (value) =>
    value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');

  const applyInlineMarkdown = (value) => {
    let text = escapeHtml(value);
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    return text;
  };

  const textToHtml = (value) => {
    if (!value?.trim()) {
      return '<p><br/></p>';
    }
    return value
      .split(/\n{2,}/)
      .map((p) => `<p>${applyInlineMarkdown(p).replace(/\n/g, '<br/>')}</p>`)
      .join('');
  };

  const [html, setHtml] = useState(textToHtml(coverLetterText || ''));
  const [copied, setCopied] = useState(false);
  const editorRef = useRef(null);

  const getPlainText = () => {
    return editorRef.current?.innerText?.trim() || '';
  };

  const exec = (command, value = null) => {
    editorRef.current?.focus();
    document.execCommand(command, false, value);
    setHtml(editorRef.current?.innerHTML || '');
  };

  const handleCopy = async () => {
    const text = getPlainText();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      if (editorRef.current && selection) {
        range.selectNodeContents(editorRef.current);
        selection.removeAllRanges();
        selection.addRange(range);
        document.execCommand('copy');
        selection.removeAllRanges();
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2500);
    }
  };

  const handleDownload = () => {
    const wordHtml = `
      <html xmlns:o="urn:schemas-microsoft-com:office:office"
            xmlns:w="urn:schemas-microsoft-com:office:word"
            xmlns="http://www.w3.org/TR/REC-html40">
      <head>
        <meta charset="utf-8" />
        <title>Cover Letter</title>
        <style>
          body { font-family: Georgia, 'Times New Roman', serif; font-size: 12pt; line-height: 1.6; margin: 36pt; }
          p { margin: 0 0 12pt 0; }
          ul, ol { margin: 0 0 12pt 24pt; }
        </style>
      </head>
      <body>${html}</body>
      </html>
    `;

    const blob = new Blob([wordHtml], { type: 'application/msword;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Cover_Letter_${(jobTitle || 'Position').replace(/\s+/g, '_')}.doc`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const plainTextFromHtml = html
    .replace(/<br\s*\/?\s*>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  const wordCount = plainTextFromHtml ? plainTextFromHtml.split(/\s+/).length : 0;

  return (
    <div className="card-elevated animate-fade-in-up" style={{ padding: '0' }}>
      {/* Header */}
      <div
        className="flex items-center justify-between px-6 py-4"
        style={{ borderBottom: '1px solid var(--color-border-subtle)' }}
      >
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
            style={{ background: 'rgba(139,92,246,0.15)', border: '1px solid rgba(139,92,246,0.25)' }}
          >
            <FileText className="w-4 h-4" style={{ color: '#8b5cf6' }} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-bold text-base" style={{ color: 'var(--color-text-primary)', fontFamily: 'var(--font-display)' }}>
                Cover Letter Draft
              </h2>
              <span className="badge badge-brand text-xs">
                <Sparkles className="w-2.5 h-2.5" /> AI Generated
              </span>
            </div>
            <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
              {company ? `For ${company}` : 'Click inside to edit'}
              {jobTitle ? ` · ${jobTitle}` : ''}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="btn btn-secondary btn-sm"
            id="cover-letter-copy-btn"
          >
            {copied
              ? <><CheckCheck className="w-4 h-4" /> Copied!</>
              : <><Copy className="w-4 h-4" /> Copy</>
            }
          </button>
          <button
            onClick={handleDownload}
            className="btn btn-primary btn-sm"
            id="cover-letter-download-btn"
          >
            <Download className="w-4 h-4" />
            Download Word
          </button>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="px-6 py-3 flex items-center gap-2"
        style={{ background: 'rgba(139,92,246,0.06)', borderBottom: '1px solid var(--color-border-subtle)' }}
      >
        <span className="text-sm">✏️</span>
        <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          <strong style={{ color: 'var(--color-text-secondary)' }}>This is an AI-generated first draft.</strong>{' '}
          Please review, personalize, and edit it to authentically represent you before sending.
        </p>
      </div>

      {/* Formatting toolbar */}
      <div className="px-4 py-2 flex flex-wrap gap-2" style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('bold')}><Bold className="w-4 h-4" /></button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('italic')}><Italic className="w-4 h-4" /></button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('underline')}><Underline className="w-4 h-4" /></button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('insertUnorderedList')}><List className="w-4 h-4" /></button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('insertOrderedList')}><ListOrdered className="w-4 h-4" /></button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('justifyLeft')}><AlignLeft className="w-4 h-4" /></button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('justifyCenter')}><AlignCenter className="w-4 h-4" /></button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => exec('justifyRight')}><AlignRight className="w-4 h-4" /></button>
      </div>

      {/* Editor area */}
      <div className="relative">
        <div
          ref={editorRef}
          id="cover-letter-textarea"
          contentEditable
          suppressContentEditableWarning
          onInput={(e) => setHtml(e.currentTarget.innerHTML)}
          dangerouslySetInnerHTML={{ __html: html }}
          className="input"
          style={{
            borderRadius: 0,
            border: 'none',
            minHeight: '480px',
            fontFamily: 'Georgia, "Times New Roman", serif',
            fontSize: '0.9375rem',
            lineHeight: '1.8',
            padding: '1.5rem',
            background: 'var(--color-bg-input)',
            color: 'var(--color-text-primary)',
            overflowY: 'auto',
          }}
          aria-label="Cover letter editor"
        />
      </div>

      {/* Footer */}
      <div
        className="flex items-center justify-between px-6 py-3"
        style={{ borderTop: '1px solid var(--color-border-subtle)' }}
      >
        <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          {wordCount} words
        </p>
        <p className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
          Changes are local only — not auto-saved
        </p>
      </div>
    </div>
  );
}
