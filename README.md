# PDF Outline Challenge — Solution

This repo delivers an **offline, CPU‑only** pipeline that extracts the
document title plus H1/H2/H3 headings from every PDF placed in
`/app/input`, producing `<name>.json` files in `/app/output`.

---

## Build

```bash
docker build --platform linux/amd64 -t pdfoutline.challenge .
