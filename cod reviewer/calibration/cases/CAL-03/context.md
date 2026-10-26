Both files train a binary head with BCEWithLogits plus Dice. Audit
the total loss formula and its weights; the optimizer and schedule
are outside this case.
