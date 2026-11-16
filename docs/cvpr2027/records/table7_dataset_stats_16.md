# Dataset statistics for the 16-dataset working subset

Counts and image dimensions are read from `data/sacm_3shot_seed0/protocol_manifest.csv`. `test` is the number of test images; `test_split` records the protocol split name. Unseen/Novel datasets have no training or validation rows because the 3-shot training source is Seen.

| Dataset | Role | Domain | Train | Val | Test | Test split | Width | Height | Mean FG ratio |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| CHASEDB1 | Seen | fundus | 3 | 1 | 20 | seen_test | 999–999 | 960–960 | 0.0704 |
| CHCrack5k | Unseen | crack | 0 | 0 | 990 | unseen_test | 480–480 | 480–480 | 0.0177 |
| CREMI | Seen | EM neuron | 3 | 1 | 123 | seen_test | 1250–1250 | 1250–1250 | 0.0267 |
| CrackTree200 | Seen | crack | 3 | 1 | 202 | seen_test | 800–800 | 600–600 | 0.0032 |
| DCA1 | Seen | coronary angiography | 3 | 1 | 130 | seen_test | 300–300 | 300–300 | 0.0545 |
| DRIVE | Seen | fundus | 3 | 1 | 16 | seen_test | 565–565 | 584–584 | 0.0863 |
| DeepCrack | Unseen | crack | 0 | 0 | 237 | unseen_test | 384–544 | 384–544 | 0.0433 |
| FIVES | Unseen | fundus | 0 | 0 | 200 | unseen_test | 2048–2048 | 2048–2048 | 0.0713 |
| HRF | Unseen | fundus | 0 | 0 | 45 | unseen_test | 3504–3504 | 2336–2336 | 0.0771 |
| LES-AV | Unseen | fundus | 0 | 0 | 22 | unseen_test | 1620–2196 | 1444–1958 | 0.0665 |
| OCTA-500 | Unseen | OCTA | 0 | 0 | 300 | unseen_test | 400–400 | 400–400 | 0.0922 |
| ROAD | Novel | road / aerial | 0 | 0 | 49 | novel_test | 1500–1500 | 1500–1500 | 0.0470 |
| STARE | Unseen | fundus | 0 | 0 | 20 | unseen_test | 700–700 | 605–605 | 0.0760 |
| TopoMortar | Novel | masonry / topology | 0 | 0 | 350 | novel_test | 512–512 | 512–512 | 0.2430 |
| WIRE | Novel | wire | 0 | 0 | 31 | novel_test | 500–1000 | 333–941 | 0.0583 |
| XCAD | Unseen | X-ray angiography | 0 | 0 | 126 | unseen_test | 512–512 | 512–512 | 0.0572 |
