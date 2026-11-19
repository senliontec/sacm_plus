# Parsing code-link extraction (2026-11-18)

Source: `/home/dev/mywork/project/segment/parsing` (1602 paper directories).
Primary extraction: GitHub/GitLab/Bitbucket repository URLs from `full.md`
before the References/Bibliography heading, with high/medium confidence from
nearby code context. PDF fallback: `pdftotext` over all 1734 parsing PDFs for
papers whose `full.md` yielded no main URL; this recovered 291 papers
that only expose code in a footnote/PDF URL. Matching: normalized `owner/repo`
against all `.git/config` remotes under `/home/dev/mywork/project/segment/3rd`
(330 unique remotes). No network access.

## Counts

- paper directories: 1601
- with full.md: 1601
- with at least one high/medium code URL (full.md or PDF): 845
- no code URL found: 756
- no full.md: 0
- all URLs already in 3rd: 237
- partially in 3rd: 15
- none in 3rd: 593
- unique missing repo URLs: 669
- PDF-fallback papers: 291

## Outputs

- `parsing_code_links_2026-11-18.csv` — per-paper mapping, confidence and source.
- `parsing_code_links_not_in_3rd_2026-11-18.txt` — unique missing URLs,
  one per line, ready for cloning.
- `parsing_code_links_not_in_3rd_details_2026-11-18.csv` — missing URLs with
  paper path/title, source, confidence and context.
- `parsing_pdf_code_urls_2026-11-18.json` — intermediate PDF URL scan.

## Caveats

- A paper may expose a dataset/project repo rather than the method repo.
- PDF fallback can include a GitHub URL from a citation/reference; the
  `context` column is included for manual verification.
- Exact `owner/repo` matching means a differently named fork/remote in 3rd is
  reported as missing; this is deliberate to avoid false “already available”.
- The text file contains high/medium and PDF-fallback links; verify before bulk
  cloning.
