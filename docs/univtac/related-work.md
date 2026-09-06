# Related work: tactile events, manipulation, and in-context experience

Updated 2026-09-06. This is the project's literature entry point, paired with
[BibTeX](related-work.bib) and the [research plan](research-plan.md). Citation
keys below identify complete entries in the BibTeX file. Reading depth is
explicit: **method** means the relevant full-text methods/evaluation were read;
**abstract** means only the primary abstract and bibliographic record were
checked. It does not mean an implementation was reproduced. Page numbers refer
to the linked PDF, and version differences are retained rather than combined.

The question remains whether a frozen Agent can use successful
tactile–action–outcome examples to improve autonomous native task success and
reduce trial and error. Event detection can help choose current observations
and organize examples. It is not a separate replacement project, and we make
no claim to have invented tactile change detection or event-based control.

## What the different lines of work establish

| Capability | Output / evidence needed | What it does not establish |
| --- | --- | --- |
| Contact-change detection | A measured change and its time, against an explicit reference | Slip, force, task phase, or the correct next action |
| Slip / event recognition | A semantic label evaluated against independently defined events | Transfer to another sensor, object, or simulator |
| Clip selection | Real frames selected from a stated available time interval | An online interrupt or a robot response |
| Online triggering | A causal decision, detection delay, false alarms and misses | Low-level stabilization or better Agent decisions |
| Automatic tactile control | A measured closed-loop task/control outcome | Historical-example ICL benefit |
| Tactile representation | Features evaluated on specified downstream tasks | A frozen general Agent can interpret our simulated images |
| In-context manipulation | Test-time demonstrations influence actions without weight updates | That historical touch adds value over matched visual examples |

For our A/B/C comparison, **current vision, bilateral touch, proprioception,
operation history, executor, and budget policy remain the same**. A has no
demonstrations; B has visual–action histories; C adds historical touch to the
same histories. Improvements from a new detector, controller, or reasoning
frequency need their own comparisons; they cannot be attributed to ICL.

## Contact changes and incipient slip

### Statistical change detection before image-based touch

**Brian Eberman and J. Kenneth Salisbury. Application of Change Detection to
Dynamic Contact Sensing. IJRR 13(5):369–394, 1994.** `eberman1994change`.
[Publisher](https://doi.org/10.1177/027836499401300501).
Reading: **1994 abstract; method in the 1993 MIT AIM-1421 predecessor**, not
verified identical to the final article. The [MIT report PDF](https://dspace.mit.edu/server/api/core/bitstreams/8234a48c-a8ee-4bca-bd19-2e63ec39871f/content)
is cited separately as `eberman1993report`, not counted as another contribution.

The report models high-rate strain-bridge signals with changes in mean,
variance, impact structure, or autoregressive dynamics. Sections 4–5, pp. 6–15,
compare candidate change points using generalized likelihood ratios and an MDL
complexity penalty; Page–Hinkley appears as a special case. Evaluation includes
Gaussian simulations, impacts, and texture transitions. Section 7 explicitly
says the full algorithm did not run in real time. Retrospective change-point
localization accuracy is not alarm latency.

**Design implication:** separate change estimation, detection time, and semantic
interpretation; report the false-alarm/delay tradeoff. Its force-signal models
and kilohertz sampling do not directly apply to RGB frames. We may test such
statistics on image-derived features later; this is a proposed transfer, not
an adopted detector or proof of slip recognition.

### GelSight: spatial deformation, relative motion, and partial slip

**Wenzhen Yuan, Rui Li, Mandayam A. Srinivasan, and Edward H. Adelson.
Measurement of Shear and Slip with a GelSight Tactile Sensor. ICRA, 304–311,
2015.** `yuan2015shear`. Reading: **method**.
[Author PDF](https://people.csail.mit.edu/yuan_wz/GelSight1/ICRA15_2740_FI.pdf),
[DOI](https://doi.org/10.1109/ICRA.2015.7139016).

Sections II–III track surface markers and interpolate their displacement.
Shear, torsion, and peripheral partial slip produce different deformation
patterns. Section III.B, p. 5, uses displacement-magnitude entropy
`H(X) = −∫ p(x) log p(x) dx` as a slip-related feature. Evaluation uses controlled
indenters, an ATI Nano17, and manually pulled can/key examples; the reported
entropy interval is setup-specific. No independent alarm-delay or false-alarm
benchmark was found.

**Design implication:** local nonuniform motion is a better candidate to examine
than interpreting all marker motion as slip. The gel mechanics, contact shape,
and quasi-static loading differ from our simulation. Marker displacement is not
automatically calibrated force, and the paper also discusses comparing object
texture/geometry motion with marker motion. It motivates a feature candidate,
not a ready-made threshold for UniVTAC.

**Siyuan Dong, Wenzhen Yuan, and Edward H. Adelson. Improved GelSight Tactile
Sensor for Measuring Geometry and Slip. IROS, 137–144, 2017.**
`dong2017gelsight`. Reading: **method in arXiv:1708.00922; publication metadata
verified**. [Paper](https://arxiv.org/abs/1708.00922),
[DOI](https://doi.org/10.1109/IROS.2017.8202149).

Sections III–V combine improved illumination/reflective skin with three slip
cues: object-texture motion relative to markers, peripheral versus global
marker displacement, and loss of contact area. Section VI evaluates everyday
objects and closed-loop regrasping. Its “border” cases are ambiguous judgments,
not a conventional third class to average into a classification score.

**Important implementation caveat:** PDF p. 6 Eq. (1) defines the peripheral/global
maximum displacement ratio. The text says the ratio exceeds the threshold;
Fig. 8 says it falls below it. Do not silently choose an inequality and attribute
it to an unambiguous published algorithm.

**Design implication:** compare multiple observable cues and retain weak-signal
failures. Texture, contact-area estimation, and smooth/rotating-object failures
limit transfer to marker-only processing. Regrasp control and its success are
not evidence for our Agent's example use.

### GelSlim: residual motion relative to a sticking center

**Siyuan Dong, Daolin Ma, Elliott Donlon, and Alberto Rodriguez. Maintaining
Grasps within Slipping Bounds by Monitoring Incipient Slip. ICRA, 3818–3824,
2019.** `dong2019incipient`.
[DOI](https://doi.org/10.1109/ICRA.2019.8793538).
Reading: **method in the 2018 preprint**, whose title uses singular *Bound*;
[arXiv:1810.13381](https://arxiv.org/abs/1810.13381). These are one contribution.

Section III, PDF pp. 2–3, estimates planar rigid motion from central markers,
predicts the rest of the contact's motion, and thresholds the residual slip
field. Eq. (1) defines the rigid-motion relation; the reference is updated
after detection. The assumptions include an approximately rigid object and a
center that still sticks. Although two GelSlim sensors are mounted, detection
uses one side. Section IV reports 30 Hz capture and about 24 Hz processing,
with marker localization the main cost. Evaluation covers manually perturbed
objects and screw/unscrew control; the 2 false alarms and 31 misses among 240
trials are **all-trial fractions**, not conditional FPR/FNR.

**Design implication:** test local residuals and track feature quality before
assigning slip labels. Our input may lack the cloth-texture contact mask or a
sticking center. Bilateral fusion and online delay remain unvalidated; do not
import the paper's automatic force-increase behavior into the Agent executor.

### FingerVision: event classes from marker sequences

**Yazhan Zhang, Weihao Yuan, Zicheng Kan, and Michael Yu Wang. Towards Learning
to Detect and Predict Contact Events on Vision-based Tactile Sensors.
PMLR 100:1395–1404, 2020; conference CoRL 2019.** `zhang2020events`.
[Publisher and BibTeX](https://proceedings.mlr.press/v100/zhang20b.html),
[PDF](https://proceedings.mlr.press/v100/zhang20b/zhang20b.pdf).
Reading: **method**. arXiv:1910.03973 is the same work.

Sections 3–4 convert tracked markers to `30×30×2` displacement fields. A 30 Hz,
one-second window is subsampled to 15 frames. An LSTM recognizes translational
slip, rotational slip, rolling, stable contact, no contact, contact formation,
and contact loss. A separate ConvLSTM/PixelMotionNet predicts future fields.
Human-keypress labels have roughly 100 ms uncertainty. Section 5 reports a
random train/validation split, not an independent held-out test; the predictor
excludes some event classes and struggles with rapid contact changes.

**Design implication:** this is a relevant learned software-event reference for
our marker streams and event vocabulary. It requires training and trustworthy
temporal labels. Neither the seven labels nor generated future fields are
available truths for our Agent. Grasp-control improvement does not measure
tactile ICL. The sequence window and inference time must both enter latency
accounting.

**Akihiko Yamaguchi and Christopher G. Atkeson. Tactile Behaviors with the
Vision-Based Tactile Sensor FingerVision. IJHR 16(3):1940002, 2019.**
`yamaguchi2019fingervision`. Reading: **publisher abstract and metadata**.
[Publisher](https://doi.org/10.1142/S0219843619400024),
[author's sensor/publication entry](https://www.cs.cmu.edu/~cga/skin/).

Transparent dotted elastic skin permits both deformation sensing and seeing
the object through the skin. The work studies tactile behaviors using these
complementary channels. **Design implication:** distinguish marker motion from
visible object motion; optical flow over an object is not necessarily gel
deformation. The transparent-skin observation assumption differs from our
GelSight-style rendering. Detailed behavior metrics and the relation to the
2017 *Implementing Tactile Behaviors Using FingerVision* paper remain to be
checked; no timing or accuracy claim is imported here.

### Calibration and hardware-event boundaries

**Karl Van Wyk and Joe Falco. Slip Detection: Analysis and Calibration of
Univariate Tactile Signals. arXiv:1806.10451, 2018.** `vanwyk2018slip`.
[NIST record](https://www.nist.gov/publications/slip-detection-analysis-and-calibration-univariate-tactile-signals),
[PDF](https://arxiv.org/pdf/1806.10451). Reading: **method**.

Sections II–IV study Nano17, OptoForce20, and BioTac SP signals at 850–1000 Hz.
Eq. (1) forms a scalar temporal difference of signal norms; BioTac supplies a
high-pass pressure channel. A small LSTM classifies slip. Sections V–VI vary
window size, sample rate, material, speed, and sensor instance. Non-slip data
include motion and pressing, exposing vibration-related false alarms.

**Design implication:** include commanded motion, pressing, and device variation
in eventual detector validation; record window duration separately from compute
time. OptoForce's optical force signal is not an RGB video. Frequency bands and
sample counts cannot transfer unchanged. The differently titled ICRA 2018
paper *Calibration and Analysis of Tactile Sensors as Slip Detectors*
([NIST](https://www.nist.gov/publications/calibration-and-analysis-tactile-sensors-slip-detectors),
DOI 10.1109/ICRA.2018.8461117) is linked as related provenance; its exact overlap
is not yet checked, so it is not counted as an additional verified method or
used as the DOI for the arXiv title.

**Amin Rigi, Fariborz Baghaei Naeini, Dimitrios Makris, and Yahya Zweiri.
A Novel Event-Based Incipient Slip Detection Using Dynamic Active-Pixel Vision
Sensor (DAVIS). Sensors 18(2):333, 2018.** `rigi2018davis`.
[Publisher](https://www.mdpi.com/1424-8220/18/2/333),
[PDF](https://mdpi-res.com/d_attachment/sensors/sensors-18-00333/article_deploy/sensors-18-00333.pdf).
Reading: **method**.

Section 3 aggregates asynchronous positive/negative brightness events into
10 ms images, filters regions, and compares their areas to distinguish slip
from vibration. Section 4 uses a synchronized 1000 FPS camera; the operational
incipient-slip reference is contact-area loss during release. It should not be
silently redefined as measured tangential object displacement. Evaluation uses
37 experiments on five objects and 20 ms scoring intervals.

**Design implication:** explicitly define the event and latency reference, and
report false alarms/misses. DAVIS hardware emits asynchronous pixel events;
ordinary `rgb_marker` frames do not. Its reported delay cannot be transferred
to our camera/render/MCP/Agent pipeline. A qualitative “stress map” is not
calibrated force, and the study's vibration scope excludes several robot/camera
vibration sources. This is a hardware comparison, not an implementation recipe
requiring us to buy an event camera.

### Lightweight marker statistics revisited

**Xiaohai Hu, Aparajit Venkatesh, Yusen Wan, Guiliang Zheng, Neel Jawale,
Navneet Kaur, Xu Chen, and Paul Birkmeyer. Learning to detect slip through
tactile estimation of the contact force field and its entropy properties.
Mechatronics 104:103258, 2024.** `hu2024entropy`.
[DOI](https://doi.org/10.1016/j.mechatronics.2024.103258),
[author-hosted journal PDF](https://faculty.washington.edu/chx/sample_publication/hu-grasping-journal-2024/paper.pdf).
Reading: **method, journal and arXiv v4**. The shorter-title
[arXiv:2303.00935](https://arxiv.org/abs/2303.00935) began in 2023; one contribution.

Journal §3 tracks 63 GelSight Mini markers at 25 Hz, using one of two mounted
sensors. Eqs. (1)–(8) construct mean displacement velocity, magnitude-histogram
entropy, and entropy rate. Shallow classifiers are evaluated across objects;
§5 includes leave-one-object-out tests and grip control. Novel-object results
are weaker than the headline pooled accuracy. The formal version discretizes
0–3 mm into 30 bins; do not copy the preprint's 15-value description.

**Design implication:** entropy plus its temporal change is a compact candidate
feature, without needing an LLM caption. Tracking, scale calibration, reference
choice, and object distribution still matter. The method is not a calibrated
RGB-to-force estimator or validated bilateral detector; its PD grip controller
is separate from the feature proposal.

### What the reported timing and error numbers mean

These are heterogeneous experiments, not a detector leaderboard. A missing
measurement is **unreported**, not zero.

| Work | Reported timing / errors | Limits for our choice |
| --- | --- | --- |
| Eberman report | False-alarm/delay analysis; full implementation not real time | Retrospective onset localization is not alarm delay |
| Yuan 2015 | No event-level delay/FPR/FNR found | Quasi-static experiments do not establish fast feedback |
| Dong 2017 | Video-level judgments and regrasp trials | Ambiguous border cases; no onset-to-alarm measurement |
| Dong 2019 | About 24 Hz processing; 2/240 false alarms, 31/240 misses | Throughput and all-trial fractions, not delay or conditional rates |
| Zhang 2020 | 9.4 ms LSTM forward pass; roughly 100 ms labeling uncertainty | Neither includes the full observation window and robot response |
| Van Wyk 2018 | 50-sample windows span about 50–59 ms; inference under 0.3 ms/window | High-rate scalar sensors, not image extraction cost |
| Rigi 2018 | Mean onset-to-first-TP 44.1 ms; precision 0.70, sensitivity 0.85 | Hardware events and paper-specific area-loss labels |
| Hu 2024 | Classifier inference 0.29–0.94 ms; 25 Hz sensing/control | Excludes complete image-feature processing and onset-delay distribution |

None of these establishes our simulation/render latency, Agent response time,
context cost, or the task-level cost of false alarms. Future measurement should
separate capture interval, feature compute, persistence/window delay, delivery,
Agent inference, and execution; report event misses and false alarms per unit
simulated time, not only balanced-window accuracy.

## Tactile representations, foundation policies, and the benchmark

The entries below recover the requested references without treating learned
policies, tactile-language QA, and historical-example ICL as interchangeable.
Full author lists and stable identifiers are in [BibTeX](related-work.bib).

| Work and verified citation | Problem, input, method, evaluation; reading depth | Project influence and boundary |
| --- | --- | --- |
| Chengbo Yuan et al., **FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation**, arXiv:2606.13102v2, 2026. `yuan2026ftp1` ([source](https://arxiv.org/abs/2606.13102v2)) | Heterogeneous image/array/state tactile encoders feed shared morphology-aware tactile tokens; pretrained manipulation policy, evaluated for sensor/embodiment transfer. **Abstract**; precise evaluation protocol remains to be read. | Existing FTP-1 source distribution is benchmark/runtime provenance. Cross-sensor learning motivates caution about sensor mismatch; it is not a frozen Codex ICL result. |
| Baijun Chen et al., **UniVTAC: A Unified Simulation Platform for Visuo-Tactile Manipulation Data Generation, Learning, and Benchmarking**, arXiv:2602.10093v1, 2026. `chen2026univtac` ([source](https://arxiv.org/abs/2602.10093v1)) | Simulation data, three supported sensor types, learned visuotactile encoder, eight-task benchmark and real-world evaluation. **Abstract**. | Recovered platform paper, distinct from FTP-1's policy paper. Our native task/checker and image provenance come from the implementation. Current Isaac51 results are not direct reproductions of legacy FTP-1/Isaac4.5 numbers. |
| Jianyi Zhou et al., **TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation**, arXiv:2607.07287v2, 2026. `zhou2026touchworld` ([source](https://arxiv.org/abs/2607.07287v2)) | Visual-language subtask planning, tactile subgoal prediction, nominal visuotactile action chunks, and tactile/proprioceptive residual correction; six contact-rich tasks with perturbations. **Abstract**; exact sensor/controller timing not checked. | New comparison in this index; no verified earlier adoption. Motivates separating slow reasoning from fast feedback, but trained residual control is not the current Agent executor or ICL evidence. |
| Siyu Wu et al., **Tactile-WAM: Touch-Aware World Action Model with Tactile Asymmetric Attention**, arXiv:2606.26663v3, 2026. `wu2026tactilewam` ([method](https://arxiv.org/html/2606.26663v3)) | Joint visual/tactile/action prediction; asymmetric attention and observed-change bias. §4 Eqs. (10)–(16) form a bilateral six-component proxy from temporal image differences, gradients, and divergence; evaluate UniVTAC, ManiFeel, real tasks. **Method excerpts**. | New candidate, not integrated. The gradient-aligned proxy should not be called calibrated force or verified marker flow. Observed-history gating is distinct from future-target training. Full-model gains do not isolate the change bias or prove an event detector/ICL benefit. Source says submitted to an RSS workshop; acceptance not verified. |
| Congsheng Xu et al., **VT-MUSE: Multimodal Unified Sequential Visuotactile Representation Learning for Manipulation**, arXiv:2608.21290v1, 2026. `xu2026vtmuse` ([method](https://arxiv.org/html/2608.21290v1)) | Temporal cross-modal alignment, masked-view learning, conditional latent representation, RGB reconstruction and tactile-depth-change supervision; frozen encoder with a separately trained action policy. §III-C/D; four simulated and four physical tasks (§IV). **Method excerpts**. | Already a representation candidate in the research plan. Supports examining short temporal history, not claiming deployment of VT-MUSE or unsupervised force/slip understanding. Privileged training targets and learned policies differ from our runtime RGB-only tactile input. Abstract and body give different headline gains; no number is imported here. |
| Samson Yu et al., **Octopi: Object Property Reasoning with Large Tactile-Language Models**, RSS 2024. `yu2024octopi` ([proceedings](https://roboticsproceedings.org/rss20/p066.html), [PDF](https://www.roboticsproceedings.org/rss20/p066.pdf)) | GelSight videos, tactile representation learning, language adaptation, and intermediate property reasoning on PhysiCLeAR. **Abstract and PDF author header**. | Historical tactile-language reference; motivates a possible interpreter. QA/property reasoning is not autonomous action success. Simulated `rgb_marker` interpretation remains unvalidated. The proceedings index reverses Kelvin Lin's name; BibTeX follows the PDF and arXiv author spelling. |
| Samson Yu, Kelvin Lin, Harold Soh, **Demonstrating the Octopi-1.5 Visual-Tactile-Language Model**, RSS 2025 demonstration paper. `yu2025octopi15` ([proceedings](https://www.roboticsproceedings.org/rss21/p058.html), [method](https://arxiv.org/html/2507.09985v1)) | Qwen2-VL-based tactile-language model, trained encoder/adaptation and simple RAG; TMI contains GelSight Mini and TAC-02, primarily using GelSight for the demonstration. §III-A selects ten frames with largest preceding-frame differences; §IV presents planned demonstration tasks. **Method**. | Existing candidate; salient-frame selection is relevant to compact inputs. Top-k selection over a supplied clip is not an online trigger. RAG object descriptions are not successful tactile–action trajectories, and neither simulated-touch transfer nor our task success has been demonstrated. Same author-name discrepancy as Octopi. |
| Jianxin Bi et al., **VLA-Touch: Enhancing Vision-Language-Action Models with Dual-Level Tactile Feedback**, arXiv:2507.17294v2, 2025. `bi2025vlatouch` ([method](https://arxiv.org/html/2507.17294v2)) | Tactile-language high-level feedback and learned diffusion/interpolant action refinement from tactile, vision and robot state; real cup/wiping/peeling tasks. **Method/training appendix excerpts**. Appendix controller training uses expert action targets; evaluation setup also describes base-VLA adaptation before tactile integration. | Historical boundary reference: “without fine-tuning the base VLA” does not mean every component was untrained. Separates semantic feedback from learned low-level control; neither is our matched-history tactile ICL. Its 8 Hz controller is not evidence for instantaneous Agent feedback. |

## Demonstrations, frozen deployment, and agent interfaces

**Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn. Learning Fine-Grained
Bimanual Manipulation with Low-Cost Hardware. RSS 2023.** `zhao2023act`.
[Proceedings](https://roboticsproceedings.org/rss19/p016.html).
Reading: **abstract**. ACT learns action sequences from teleoperated visual/robot
trajectories and evaluates fine bimanual tasks. It is imitation training, not
new demonstrations inserted into a frozen Agent's prompt. **Project role:**
action-chunking/control reference and a baseline used by neighboring tactile
papers; preserve actual timing and overlapping commands when recording actions.
No adoption of ACT or temporal ensembling in our current controller is claimed.

**Norman Di Palo and Edward Johns. Keypoint Action Tokens Enable In-Context
Imitation Learning in Robotics. RSS 2024.** `dipalo2024kat`.
[Proceedings](https://www.roboticsproceedings.org/rss20/p096.html).
Reading: **abstract**; followed from ICRT's related work. KAT converts visual
keypoints and demonstration action trajectories into tokens for an unmodified
GPT-4 Turbo, evaluating real-world manipulation with few examples.
**Project implication:** a direct frozen-language-model manipulation precedent;
example/action serialization and current-scene grounding matter. Its geometric
keypoints are a stronger preprocessing assumption than raw tactile clips. It
does not establish tactile ICL, and its detailed grounding pipeline still needs
reading before borrowing it. New comparison, not historical adoption.

**Max Fu, Huang Huang, Gaurav Datta, Lawrence Yunliang Chen, Will Panitch,
Fangchen Liu, Hui Li, Ken Goldberg. ICRT: In-Context Imitation Learning via
Next-Token Prediction. ICRA, 5937–5944, 2025.** `fu2025icrt`.
[DOI](https://doi.org/10.1109/ICRA55743.2025.11128272),
[read preprint](https://arxiv.org/html/2408.15980v2).
Reading: **method §§III–IV; publication metadata verified through DOI registry**.
The 2024 preprint omits “ICRT:” in its title and lists Letian Fu / William
Chung-Ho Panitch; do not duplicate it as another result.

ICRT trains on camera/proprioception/action trajectories, then accepts new
teleoperated trajectories without further weight updates. It represents
actions as end-effector deltas including rotation and gripper action. Its main
“new tasks” use known primitives with new objects/configurations.
**Project implication:** preserve complete observation/action tuples and test
whether demonstrations actually influence decisions. A robot-trained causal
Transformer is not an off-the-shelf Codex operator; no tactile advantage is
established. This is a new comparison, not a model we integrated.

**Vitalis Vosylius and Edward Johns. Instant Policy: In-Context Imitation
Learning via Graph Diffusion. ICLR 2025.** `vosylius2025instant`.
[Proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/5692c7dbc4abcaa50f9ce609819212e5-Paper-Conference.pdf),
[arXiv v2](https://arxiv.org/html/2411.12633v2).
Reading: **method §3.1 and evaluation/transfer excerpts**.
An offline-trained graph diffusion policy combines segmented point clouds,
gripper state, and a few demonstrations, producing SE(3) displacements and
binary gripper commands without test-time training. Evaluation includes
simulated and everyday real manipulation.
**Project implication:** explicitly relate demonstration geometry to the current
scene instead of replaying absolute targets. Point-cloud segmentation, trained
geometry priors, and known embodiment mappings are additional assumptions.
It is an ICIL comparison, not proof of raw-touch understanding or integration.

**Yitong Chen et al. ETA: A New Agentic Paradigm for Embodied Tasks.
arXiv:2608.03924v1, 2026.** `chen2026eta`.
[Primary record](https://arxiv.org/abs/2608.03924v1). Reading: **abstract plus
local architecture/code**. ETA separates Planner, Interface, and World with
one-tool-at-a-time feedback and fresh observations. OpenETA is the released
implementation. **Actual design influence:** our general tools, execution
receipts, model-visible/host-only separation, and replay. This task did not
independently check the paper's evaluation protocol; our native outcomes come
from our own runs. The paper and the `openeta-for-codex` software branch are
different citation objects.

**Letian Fu et al. CaP-X: A Framework for Benchmarking and Improving Coding
Agents for Robot Manipulation. arXiv:2603.22435v2, 2026.** `fu2026capx`.
[Primary record](https://arxiv.org/abs/2603.22435v2). Reading: **abstract; earlier
project method notes recovered, not freshly reverified in full**.
CaP-Gym/Bench study coding agents, perception/control primitives, abstraction,
multi-turn execution feedback, and visual differencing in simulated/real
manipulation. The framework includes both training-free Agent0 and a separate
RL component. **Historical influence:** visual differencing motivated compact
change reports. The old claim that CaP-X lacked this connection was corrected
in the retained review. It does not authorize tactile captions containing task
answers or renewed R0.9.19–R0.9.21 prompt ablations.

**Ziniu Hu, Ahmet Iscen, Chen Sun, Kai-Wei Chang, Yizhou Sun, David A. Ross,
Cordelia Schmid, Alireza Fathi. AVIS: Autonomous Visual Information Seeking
with Large Language Model Agent. NeurIPS 2023.** `hu2023avis`.
[Primary record](https://arxiv.org/abs/2306.08129v3).
Reading: **abstract; historical project notes**.
Planner, reasoner, working memory, human-derived transition graphs and examples
guide external-tool use for Infoseek/OK-VQA. **Historical influence:** bounded
evidence memory and information selection. Its tools gather visual knowledge;
it does not validate contact sensing, robot motion, or tactile-action ICL.

Local provenance for these historical influences is retained outside the
canonical checkout: workspace `findings.md:15`,
`refine-logs/EXPERIMENT_PLAN.md:44–46`,
`review-stage/ROSETTA_TACTILE_NEXT_STAGE_REVIEW.md:248–284`, and
`refine-logs/FINAL_PROPOSAL.md:9–53`. These locate earlier design decisions;
they are not substitutes for the primary publications above. The archived
root tracker is not a queue of experiments to resume.

## Software and equipment references — not papers

| Reference | What was inspected and what we reuse | Boundary |
| --- | --- | --- |
| **OpenMOSS/OpenETA, `openeta-for-codex` branch**. `openetaSoftware` ([repository](https://github.com/OpenMOSS/OpenETA/tree/openeta-for-codex)) | Upstream README and local [architecture](../architecture.md); [MCP server](../../tools/embodied_mcp_server.py), [Gateway](../../tools/embodied_gateway.py), synchronous worker, native images and replay | Software branch, not an additional ETA paper or automatic proof of UniVTAC control reliability |
| **robocurve/inspect-robots**. `inspectRobotsSoftware` ([repository](https://github.com/robocurve/inspect-robots)) | README and retained exploration checkout: `src/inspect_robots/{controller,rollout}.py`, agent `_tools.py`, CaP-X `_motion.py`; separation of policy calls, chunks, low-level steps, semantics and logs | Engineering comparison previously explored; no dependency/integration found in the scoped project search. Its controllers and adapters are not our validated UniVTAC backend |
| **OpenETA-UniVTAC adapters** (this repository) | [OpenEtaAgentAdapter](../../adapter/openeta_agent.py), [UnifiedSimulatorAdapter](../../sim/adapter.py), [RobotState / EnvObservation / EnvAction](../../adapter/protocol.py), [NativeController](../../sim/envs/univtac/autonomous_operation.py), [live session](../../sim/envs/univtac/autonomous_session.py) | Concrete current implementation. Robot FK/Jacobian and measured TCP feedback are distinct from hidden object geometry; LIBERO Panda grip-site assumptions must not be copied to UniVTAC's native gripper-center TCP |
| **GelSight Mini product sheet**. `gelsightMiniSheet` ([manufacturer PDF](https://www.gelsight.com/wp-content/uploads/productsheet/Mini/GS_Mini_Product_Sheet_10.07.24.pdf)) | Manufacturer equipment description; local [runtime assets](../../sim/envs/univtac/runtime.py) name GelSight Mini calibration and Franka gelpad assets | A device document, not simulated-sensor fidelity validation. No hardware rate, force calibration, or slip guarantee is inferred for our renderer |

The retained Inspect exploration is `/tmp/inspect-robots-explore-wbAvxx` on the
research machine; it is a temporary historical checkout, not a portable
dependency. Current upstream branches may evolve. Cite an actually used
release/commit when code is adopted; this index does not install or switch one.

## First implementation candidate — R1.5 segment-end implementation

R1.4 captured touch only between tools. R1.5 now records the existing native
control/render updates and selects real frames at the end of each action
segment. One unscored debug is complete; formal Agent validation is pending.
This is a simplified implementation in [tactile_history.py](../../sim/envs/univtac/tactile_history.py),
not reproduction of a cited detector: fixed-ROI integer patch matching on
160-pixel-wide images, displacement/quality scores, and labelled image-difference
fallback. It does not fit GelSlim's sticking-center rigid model, estimate force,
or train a semantic event model. Both score streams operate on the same images.
The online-trigger ideas below remain proposals: persistence, hysteresis and
refractory intervals are not implemented for this segment-end selector.

| Candidate | Required observed input | Advantage | Main risk / cost |
| --- | --- | --- | --- |
| Local image differences | Consecutive, same-side RGB frames and fixed sensor ROI | Minimal processing; transparent control | Lighting/render noise, marker movement, and physical contact changes are mixed |
| Marker / local-motion features | Same frames plus trackable markers or textured local patches | Spatially localized displacement, dispersion, and residuals; easier to inspect than captions | Tracking loss, gel reference drift, low texture, camera motion, unknown physical scale |
| Learned temporal event model | Causal feature windows and development labels with uncertain onset intervals | Can distinguish named events beyond a scalar change | Sensor/domain shift, class balance, label cost, window delay and training cost |

**Recommend a small causal local-motion change detector**, with fixed-ROI local
image difference as its simple control. First extract marker or local patch
displacements in image pixels separately for each pad; retain track-quality
indicators and robust displacement/dispersion changes. Use a trailing reference
and a threshold with short persistence, hysteresis and a refractory interval,
chosen on separate development recordings. These are ordinary candidate design
choices, not published universal constants. Compare both methods with the same
sampling, clip budget and evaluation windows. If tracks are unreliable, report
that limitation; do not replace them with actor poses or fabricated motion.

Initially name the output **tactile image/motion change**, not incipient slip.
The most useful precedents are the statistical change-time separation,
GelSlim's spatial residual idea, and FingerVision's explicit event/label
definition. Hu's entropy/rate is an optional compact feature after tracking
quality is established. A trained event vocabulary comes later if these simple
features cannot separate the events that matter in actual failures.

The online path uses only frames already captured at decision time. Keep a
small trailing buffer and emit the real pre-event frames plus the trigger
frame, with left/right labels, capture/arrival times, simulation step, action
ID, requested gripper/motion command, and measured robot state. Deliver the
selected images through the native MCP image path, not filenames or synthetic
explanations. Optional post-event frames can be appended only after they exist,
with their later availability explicit; they cannot justify an earlier alarm.
Never advance physics solely to fill a clip.

Recording frames, delivering them to the Agent, and interrupting an in-progress
motion are three separate capabilities. First recording/selection need not
change execution or reasoning frequency. Segment-end top-k peaks can use the
whole completed segment; an online alarm cannot know that a later frame will
be a larger peak. Frame cadence and dropped frames limit which fast events are
observable. Active closing/opening, commanded rotation, and contact loading
themselves cause tactile changes: retain them aligned with actions, rather than
automatically suppressing them or labeling them unintended slip. Two sides
need not change equally, so bilateral agreement is evidence to inspect, not a
mandatory veto on one-sided changes.

Future validation should use independent human-reviewed event/onset intervals,
report uncertainty and unlabelable cases, and keep labels/evaluator truth out
of runtime inputs. Measure false alarms during no-contact motion and active
gripper commands, missed short events, onset-to-alarm delay, processing time,
images/tokens delivered and task effects. A detector-selected successful
demonstration still retains real tool calls, outcomes and recovery; it does
not replace the action history with event labels. For B/C, select the history
and non-tactile timeline once and share them: C adds touch without quietly
changing the examples or giving B touch-derived annotations.

## Remaining verification and maintenance

- Obtain the 1994 Eberman journal full text before citing report-specific
  equations as if they were from that final version.
- Resolve Dong 2017's ratio inequality through code or an author correction
  before implementing that exact slip rule; verify formal/preprint method
  differences for Dong 2019 and the NIST pair when needed.
- FingerVision 2019 behavior details, KAT grounding, and the abstract-only
  foundation-policy/platform evaluations remain incompletely read. They are
  not counted as reproduced methods. No unverified venue acceptance is added
  to 2026 preprints.
- Earlier proposal names also include Tactile-VLA-CoT (2507.09160), Sparsh,
  AnyTouch, OmniVTLA/OmniVTA, ViTacFormer, TACTO, Taxim and TacEx. Their historical
  mentions are recovered as leads; exact versions, alias collisions and design
  use need primary-source checking before a formal citation is added. They do
  not block the verified entries or reactivate an archived method branch.
- Whenever a new paper is actually cited or adopted, update this entry point
  and BibTeX with the source, reading depth, version and concrete design effect.
  Keep one record per contribution, with preprint/publication relationships;
  separate software and equipment entries. No per-paper document is required.

Segment-end selection is implemented and debug-tested in R1.5. Semantic event
detection, online interruption, temporal-model integration and tactile ICL gains
remain **unimplemented or unverified**. The existing R1.4 result,
historical experiments and current A/B/C research question remain unchanged.
