# Milestone 2: one-scene action-buffer smoke check

Same pretrained SmolVLA, same bowl-placement scene and 150-action budget. Every inference predicts 50 actions. Only the execution horizon changes. No artificial delays.

| Actions used per inference | Outcome | Executed actions | Model calls | Total inference | Inference / executed action |
| --- | --- | --- | --- | --- | --- |
| 1 | SUCCESS | 78 | 78 | 24.03 s | 0.308 s |
| 10 | SUCCESS | 79 | 8 | 2.96 s | 0.037 s |
| 50 | SUCCESS | 78 | 2 | 0.58 s | 0.007 s |

K=1 discards 49 predictions and replans after each action: it is not a model trained to emit single actions. K=10/50 reuse more of each chunk, amortizing inference but delaying fresh feedback. This checks execution buffering; it does not isolate the model architecture's intrinsic chunk-generation advantage.

Simulation pauses for inference. These timings show computation cost, not real-time deadline handling. Run order is 10, 1, 50; cold-start and setup overhead are retained. One trial per condition does not establish reliability or a generally optimal horizon. Milestone 1's archived results are unchanged.
