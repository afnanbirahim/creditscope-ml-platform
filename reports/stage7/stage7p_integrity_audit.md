# Stage 7P post-evaluation integrity audit

This audit performed no model fitting, prediction generation, calibration,
threshold evaluation, or alternative-model evaluation. It read preserved files
only.

## Threshold-search discrepancy

The SHA-256 computed directly from `reports/stage6/threshold_search.csv` is:

`4f738292925c51242b34fad89d8a9c763c19014a3de3afea31f3968c3e7ac1a9`

This is 64 hexadecimal characters. The previously quoted value ending in
`e7ac1` is only 62 characters and omitted the final `a9`. Existing on-disk Stage
6 and Stage 7 artifacts already contain the correct value; the discrepancy was
limited to narrative transcription.

## Verified hashes

| Artifact | SHA-256 |
|---|---|
| Raw snapshot | `4ce6007c9eb3dbcd67e7858773566b0eb3ff140033cf67b2d8a42b3a89edfb9d` |
| Locked split | `32fb4c9ac2cdb358e3bffd145cd97e1a393c14946888d46212d0b395523e99af` |
| Stage 4 fold membership | `513fc9249a4c00be3ba67fa99ea20a7ac76b653694130cf66733ea08f97d0b92` |
| Stage 4 OOF | `c1e96b552ca7f5510f8aa9eda88db128e9cfbdd8ef30ca8535fac69646c486f9` |
| Stage 5 nested OOF | `eecccdb9ae24d747562b5914c5f54bc5838688df4019b013552d1b87217460f5` |
| Stage 6 nested OOF | `a11703e95c1fd17c05de3c2636aba4bab74bdd3d93a992140cefd0b58bbda621` |
| Stage 6 threshold search | `4f738292925c51242b34fad89d8a9c763c19014a3de3afea31f3968c3e7ac1a9` |
| Frozen Stage 6P policy | `afee2e4ea82d438fa0ba638a8b7bdadaecbce9a4ada7da1f72d66866e4313afc` |
| Stage 7 pre-access manifest | `bd989e2c1172c3822d3c80e4e033d33af9d761c76802d31f1e0d81eb267e1a01` |
| Final holdout predictions | `dc43c8f81bfd993ec6735253be3fdc572ba44deda2743d4c7b9f8dbff41dce18` |
| Final metrics | `c676b0f0ea0a02a4bfe8b681cc29903831aef83631def31c38eeb599879b124d` |
| Final confusion matrix | `6ee6ab1e4661396d2735ce3394b8e19a6c7a522a4126bbcf84e062e1375e963b` |
| Stage 7 run summary | `a48d190dacba171ec0ebac22783ed5e6462134cc80bdc3a00110f5e8985b0fb8` |

Every digest is exactly 64 hexadecimal characters.

## Final result preservation

The final metrics artifact is unchanged: ROC-AUC 0.804643, Average Precision
0.679212, precision 0.398496, recall 0.883333, F1 0.549223, balanced accuracy
0.655952, specificity 0.428571, Brier score 0.152338, log loss 0.475878,
accuracy 0.565000, TN 60, FP 80, FN 7, TP 53, total cost 115, and average cost
0.575000.

No underlying evidence, model, calibration, threshold, prediction, metric, or
confusion-matrix artifact was modified.
