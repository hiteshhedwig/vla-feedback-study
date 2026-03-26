# Attribution

- SmolVLA, its architecture, pretrained weights and processors come from
  Hugging Face LeRobot. Model:
  https://huggingface.co/lerobot/smolvla_libero .
  Model license: Apache-2.0.
- LeRobot source and its LIBERO observation conventions:
  https://github.com/huggingface/lerobot/tree/a24998f7ba3c77ea445b48c92ad15c14a50e492a .
  Copyright Hugging Face contributors; Apache-2.0. The axis-angle/image adapter
  follows these documented conventions.
- LIBERO tasks, initial states, language instructions, goal predicates and
  simulation assets:
  https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01 .
  Copyright 2023 Lifelong Robot Learning; MIT.
- Robot dynamics/control and simulation are provided by robosuite and MuJoCo,
  under their upstream licenses. This project does not implement a new robot
  controller, physics engine or neural architecture.

Complete copies of LeRobot and LIBERO licenses are retained under docs/licenses/.
Dependencies/assets remain in their upstream distributions, with their notices.
Project contributions are the small execution-mode integration, paired experiment,
evidence collection, diagnostics, reports and annotated demonstration.
