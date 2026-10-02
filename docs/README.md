# Project-page preview

Open `index.html` directly or serve `docs/` with a local static server.

The method animation and results were ported from
`ml-thesis/codes/site-playground` on 2026-10-02. This preview lives on
`feature/method-results-preview`; publishing remains a separate step.

The method uses the original page's blue/rose modality colors, neutral panels,
system fonts and blue controls. Three source frames, a waveform and a log-mel
spectrogram illustrate the data flow. The animation lasts 18 seconds and supports
pausing, continuous scrubbing, reverse seeking and reduced motion. The projected
batch histogram illustrates the Gaussian target; SIGReg itself tests characteristic
functions. Token sequences and latent positions are illustrative, not inference.
Source asset metadata is in `assets/method/provenance.json`.

The static tables preserve all rows, values and protocol notes from the playground's
`resultTables` in `app.js`: frozen attentive transfer, end-to-end fine-tuning and
cross-modal retrieval. Model names follow the current manuscript. These tables
replace the older result and retrieval tables and work without JavaScript.

Validation: five viewport widths (1440, 1024, 768, 390, 320px), exact comparison
of all 17 result rows and three protocol notes, pause/drag/resume and reverse
seeking, 32 distinct timeline updates over 32 sampled frames, existing figure
lightboxes, and tables with JavaScript disabled. No JavaScript errors or missing
local assets.

The figure caption follows the ICLR 2027 teaser caption
(`papers/iclr2027/main.tex`, Figure 1), using this page's AV-JEPA name.
The diagram has an icon-only play/pause control and scrubber, with no narration
band or replay button. On desktop the view rows, Vision Transformer/projector,
embedding panel and SIGReg panel share a vertical centerline. The projector
readout has a longer connector; modality token sequences are contiguous and
local sequences shorten as their missing modality is removed.
