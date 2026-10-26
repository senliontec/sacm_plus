The shared evaluator expects a binary mask at the original GT
resolution. Both files export the decoder output to GT size before
thresholding. Audit the export path, including interpolation.
