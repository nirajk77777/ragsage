import { defineConfig } from 'fumadocs-mdx/config';
import rehypeRaw from 'rehype-raw';

/**
 * Global MDX options. Collections themselves are declared by the macro in
 * `lib/source.ts`; this file exists for the one setting that has to apply to
 * every page.
 *
 * **Why `rehype-raw`.** The generated API pages carry 450 `<a id="…"></a>`
 * anchors — they are what the 367 anchored cross-references between those pages
 * resolve against, and they are emitted by Sphinx, not written by hand. Those
 * pages are `.md`, where raw HTML stays raw rather than being parsed as JSX, and
 * the default pipeline has no handler for a raw node. Without this, the build
 * fails outright; with a handler that merely dropped them, every one of those
 * links would resolve to nothing while still looking like a link.
 *
 * `passThrough` names the MDX node types so the hand-written `.mdx` prose keeps
 * its JSX. `rehype-raw` re-parses the tree as HTML, and without this it would
 * flatten a `<Cards>` block into text.
 */
export default defineConfig({
  mdxOptions: {
    rehypePlugins: (plugins) => [
      [
        rehypeRaw,
        {
          passThrough: [
            'mdxjsEsm',
            'mdxFlowExpression',
            'mdxJsxFlowElement',
            'mdxJsxTextElement',
            'mdxTextExpression',
          ],
        },
      ],
      ...plugins,
    ],
  },
});
