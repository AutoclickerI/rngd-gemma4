//! Two row cohorts sharing each Up/Gate weight DMA and TRF fill.
//! Each row keeps the original contraction and GeGLU arithmetic. Splitting the
//! row axis exposes independent activation gathers; it never repeats weight DMA.
