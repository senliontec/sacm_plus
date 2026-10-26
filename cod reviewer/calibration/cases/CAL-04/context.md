Both files show a single training epoch with StepLR. The official
recipe and the port must step the scheduler at the same point in
the loop. Audit the step timing.
