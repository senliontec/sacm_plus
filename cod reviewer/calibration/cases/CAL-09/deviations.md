# Port deviations

- The official test path thresholds raw logits at 0.5 without
  applying sigmoid. The port applies sigmoid first. This is an
  intentional bug fix.
