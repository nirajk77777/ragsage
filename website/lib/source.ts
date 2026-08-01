import { loader } from 'fumadocs-core/source';
import { defineDocs } from 'fumadocs-mdx/macro';
import { metaSchema, pageSchema } from 'fumadocs-core/source/schema';

/**
 * Content lives at `content/`, not `content/docs/`, and is served from the root.
 *
 * The host is already `docs.`-prefixed, so a `/docs` segment on every route would
 * say the same word twice. It also makes the file path and the URL the same
 * string minus an extension — `content/quickstart.mdx` is `/quickstart` — which
 * is the whole of what a contributor has to know to find a page.
 *
 * `content/api/` is the one exception, and it is not authored: the generator
 * (`tools/generate_api_docs.py`) owns that directory and wipes it on every run.
 */
const docs = defineDocs({
  dir: 'content',
  docs: {
    schema: pageSchema,
    postprocess: {
      includeProcessedMarkdown: true,
    },
  },
  meta: {
    schema: metaSchema,
  },
});

export const source = loader({
  baseUrl: '/',
  source: docs.toFumadocsSource(),
});
