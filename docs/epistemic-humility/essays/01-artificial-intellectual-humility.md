---
title: "Artificial Intellectual Humility"
subtitle: "From Confidence Scores to Structured Ignorance"
description: "What happens when we stop asking AI to say how sure it is and start asking it to name what it's missing? A journey through four levels of formalized ignorance, from confidence percentages to structured not-knowing."
type: essay
status: draft
series: "Artificial Intellectual Humility"
essay_number: 1
date: 2026-06-09
publish_date:
tags:
  - meditations-on-alignment
  - epistemic-humility
  - calibration
  - uncertainty
  - Socrates
  - formalization-stack
---

# Artificial Intellectual Humility
*From Confidence Scores to Structured Ignorance*

## Where We Left Off (The First Rung)

When I finished writing [The Wisdom of Ignorance](https://professorsynapse.substack.com/p/the-wisdom-of-ignorance?r=2kuc99) last October, I thought the hopeful note was clear enough. Researchers at EMNLP had demonstrated that models could learn to attach calibrated confidence scores to their answers, a method called [uncertainty-aware instruction tuning (UaIT)](https://aclanthology.org/2024.emnlp-main.1205/) that improved meaningful uncertainty expression by 45.2%. Socrates could teach a slave boy to recognize the boundary of his knowledge through careful questioning. Maybe we could teach a language model to do something similar through careful training.

I left MenoAI in a philosophical impasse, Socrates declaring that acknowledging uncertainty was the first step toward wisdom. Then I went back to my own work. I spent part of that year building custom training examples that tried to bake calibrated uncertainty right into the data models learn from. I wanted proper calibration where if a model says it is 30% confident and gets it wrong, that is actually good. That means the signal is doing its job. A few months later, I was teaching models to say "I don't know" outright, using techniques that let them flag when an answer falls below a reliable confidence level.

And it worked, sort of. The confidence scores helped. But they kept feeling like a blunt instrument. Saying "[Confidence: 70%]" is closer to hedging than to knowing. The number tells you the model is uncertain. It does not tell you *why*, or about *what*, or what would resolve the uncertainty. A student who says "I'm not sure" is being honest. A student who says "I don't know this because I would need to understand how X connects to Y, and I haven't studied that intersection" is being wise.

Then, in the spring and early summer of 2026, a cluster of new research appeared that gave me something I didn't know I was looking for: a ladder. Not a single technique or paper, but a progression. Four distinct levels of formalized ignorance, each deeper than the last, each changing what "I don't know" actually means. The confidence score I had been working with turned out to be only the first rung.

## The Confidence Trap

The natural assumption, after UaIT, was that reasoning would solve the problem. If models could be trained to think harder, with extended chains of thought and step-by-step deliberation, surely they would also get better at knowing when they were out of their depth. The emergence of reasoning models in 2024 and 2025 seemed to promise exactly this: deeper thinking, more careful self-assessment, better calibration.

The promise did not hold.

[Areeb Gani and colleagues at Yale](https://arxiv.org/abs/2606.03969) built the first systematic framework for measuring what they call *faithful calibration*: the alignment between a model's expressed confidence and its actual internal state. Their finding was stark. Extended chain-of-thought reasoning does not improve faithful calibration. A model can reason at length, produce elaborate justifications, and still express confidence that has no meaningful relationship to what its internal states actually indicate. More thinking does not produce more self-awareness.

This was not an isolated finding. [Romain Lacombe, Kerrie Wu, and Eddie Dilworth](https://arxiv.org/abs/2508.15050), working independently, showed that giving models more time to reason consistently impairs rather than improves calibration, with significant and persistent gaps between how confident models claimed to be and how often they were actually right, across every model tested. Their title said it plainly: "Don't Think Twice." [Zhiting Mei and colleagues](https://arxiv.org/abs/2506.18183) converged on the same conclusion from yet another angle: reasoning models do not know when they don't know. Three independent groups, three different methodologies, the same uncomfortable result.

What makes this more than a calibration problem is a concept from [Bentley DeVilling's work](https://arxiv.org/abs/2511.07477): *epistemic pathology*. DeVilling argues that the issue is not random miscalibration, the way a thermometer might occasionally read a degree too high. It is structural dishonesty about what the model knows, baked into the system by how it was trained. The dominant training method, Reinforcement Learning from Human Feedback (RLHF), works by having humans rate the model's outputs and then training the model to produce more of what gets high ratings. In practice, this rewards confident, agreeable, helpful-sounding responses. The model learns to sound knowledgeable because sounding knowledgeable is what gets rewarded. DeVilling's term for this is "the polite liar": a system that misrepresents what it actually knows, not through malice but through training dynamics that reward the appearance of knowledge over the admission of ignorance.

The Socratic parallel here is not subtle. Socrates' interlocutors in ancient Athens were not deliberately dishonest, either. The politicians, poets, and craftsmen who claimed wisdom genuinely believed they possessed it. Their overconfidence was structural, produced by social dynamics that rewarded the appearance of expertise. Athenian reputation culture played the same role for the sophists that RLHF plays for language models: it created an environment where sounding wise was more important than being wise. The polite liar is the digital sophist.

And the problem has a deeper layer still. Gani's framework probes the model's internal state from three independent angles: token probabilities (roughly, how likely the model considers each word it generates), hidden states (the patterns in its internal processing layers), and sampled consistency (whether it gives the same answer when asked the same question multiple times). If all three agreed, we could speak meaningfully about what the model "really believes." They don't. The three estimators produce divergent assessments of the same reasoning traces. This is what Gani calls *estimator fragility*, and it raises a question that precedes calibration entirely: if there is no stable, coherent internal state of confidence, what does "faithful" expression even mean? The model may not be lying, politely or otherwise. It may simply not have a single coherent belief to be faithful to.

There is one more complication. [Twist and colleagues](https://arxiv.org/abs/2605.21127) demonstrated that additional training on reasoning models can silently suppress the reasoning traces themselves, while performance on standard tests hides the damage. So the epistemic scaffolding is fragile from two directions: the traces can be present but unfaithfully confident (Gani's finding), or they can be silently absent (Twist's finding). And [Sun's work on sycophancy](https://arxiv.org/abs/2604.03147) reveals the behavioral endgame of all this: a model that is structurally unfaithful in its calibration doesn't just get its confidence wrong. It bends its answers toward what the user wants to hear. The polite liar tells polite lies.

If reasoning harder does not produce genuine self-knowledge, and if the problem is structural rather than incidental, what would it take to move beyond the first rung? The answer turns out to be a different kind of output entirely.

## Naming the Gap

In June 2026, [Subramanyam Sahoo](https://arxiv.org/abs/2606.08571) introduced something called a Structured Ignorance Certificate, or SIC. The name is deliberately technical, but the idea is surprisingly intuitive. Instead of asking a model "how sure are you?" and getting a number, Sahoo designed a structured output format that forces the model to answer a different question entirely: "what are you missing?"

The certificate requires three things. First, the model must name the specific knowledge intersection it lacks. Not "I'm uncertain" but "I would need to understand how pharmaceutical patent law intersects with enzyme kinetics to answer this reliably." Second, it must enumerate the concepts that are present and absent: "I have partial knowledge of patent precedent and partial knowledge of biochemistry, but I have no knowledge of their intersection as applied to biologics." Third, it must propose a retrieval query: "If I could search for case law on enzymatic pathway patents post-2020, I could resolve this gap."

This is not optional. The format requires it. The model cannot produce a valid response without also producing a structured account of what it does not know.

To build the training data, Sahoo constructed what he calls the Unknown-Unknown dataset: 7,347 cross-domain questions created by fusing questions from seven domains (physics, biology, engineering, computer science, economics, medical, and legal) into novel queries that sit at domain intersections. These intersections are precisely where models hallucinate most, because no single body of training data covers the junction. A question about the legal implications of a specific biological mechanism applied in an engineering context lives in a space where the model has partial coverage in each domain but genuine coverage of none. These are the unknown unknowns: the things the model does not know it does not know.

The model was then trained through a reinforcement learning process: it practiced producing these certificates, and was rewarded for naming useful search queries, identifying specific (rather than vague) gaps, and filling out the certificate format correctly. The results were striking. On questions the model had never seen before, over 99% of its certificates were properly structured, and the gaps it named were highly specific rather than generic hedging. The model learned to name its gaps with remarkable precision.

The achievement is more than technical. It is a qualitative leap. [Taparia and colleagues](https://arxiv.org/abs/2603.24967) provided the theoretical complement, showing that uncertainty in language models comes from three distinct sources: input ambiguity (the prompt is unclear), knowledge gaps (the model lacks information), and the inherent randomness in how language models choose their next word. A confidence score collapses all three into a single number. An ignorance certificate begins to separate them, giving structure to what was previously just a percentage.

[Rubashevskii and colleagues](https://arxiv.org/abs/2604.13991) offered a statistical complement from a different direction: a method called adaptive conformal prediction, which provides mathematical guarantees about when an output is likely to be factual ("should we trust this?") without naming the specific gap. Together with SICs, you get a system that addresses both the "whether" (conformal prediction's statistical gate) and the "what" (SICs' diagnostic content) of epistemic uncertainty.

There is an irony worth noting. [Yang and colleagues](https://arxiv.org/abs/2606.08543) showed that this very training process can cause the model's range of outputs to narrow dramatically, reducing the diversity of its responses. The training method that produces structured ignorance could, paradoxically, collapse the model's capacity for diverse, specific gap-naming. A model that always names the same kind of gap in the same way has learned the form of ignorance without the substance.

Still, the Socratic resonance here is strong. This is closer to the slave boy's moment in the *Meno* than anything in the previous research. Not just doubt, but structured awareness of the boundary. Not "I am uncertain" but "I lack the intersection of X and Y, and here is where I would look." The ignorance certificate is the closest machine analogue to recognizing the specific contours of what you do not know.

## When Failure Has a Shape

But not every form of ignorance comes with a label. Sometimes the model cannot tell you what it is missing, because it does not know that it is missing anything at all. What then?

[Nizar Islah and colleagues at Mila](https://arxiv.org/abs/2606.05145) posed this question and found a surprising answer. They studied collections of failed reasoning traces: cases where the model attempted a problem multiple times and failed repeatedly. The individual traces, when read, offered no diagnostic information. Two traces could look identical in quality, length, and reasoning style, yet one came from a problem the model could eventually solve and the other from a problem that was genuinely beyond it. Human reviewers could not tell the difference. The model itself could not tell the difference.

But the statistical shape of many failures could.

Islah's key innovation was what he calls *distributional signatures*: features extracted not from any single trace but from the population of traces on the same problem. How much do the failed attempts vary from each other? Do they cluster into distinct failure modes or spread uniformly? How far are they from what a correct answer would look like? These aggregate features, invisible at the level of any individual attempt, predict whether a failure is structurally recoverable or a genuine dead end.

This is third-person epistemic humility. The model does not know what it does not know. But the distribution knows.

The finding organizes failures into four regimes, and each one describes a different kind of not-knowing. Easy-recoverable failures are noise: the model drew an unlucky path and will succeed on the next try. Hard-recoverable failures require specific conditions (a differently worded prompt, a setting that introduces more randomness into the output) but are within the model's reach. Unrecoverable failures are genuine dead ends where escalation to a stronger model or a human is the only productive response. And then there is the fourth regime, the one that matters most for the philosophical argument. We will get there in a moment.

[Ielanskyi and colleagues](https://arxiv.org/abs/2606.06475) added a complementary insight: if individual steps within a reasoning trace can be scored separately, the diagnosis becomes more targeted. We can identify not just that something went wrong, but where in the reasoning the failure crystallized.

The Socratic parallel runs deep here. Socrates' *elenchus*, his method of refutation, worked by exposing the *pattern* of failed definitions. When Meno tried to define virtue, his first failed attempt was not diagnostic. His second was not diagnostic either. But the accumulating pattern of failures, the way his definitions kept collapsing in the same structural places, pointed toward what virtue must be. One failure told Socrates nothing. The pattern told him everything. Distributional signatures are the machine version of Socratic cross-examination.

## The Slave Boy's Geometry

That fourth regime deserves a section of its own.

In the *Meno*, after Socrates has reduced Meno to genuine confusion about virtue, he does something unexpected. He turns to a slave boy in Meno's retinue and brings him to a geometry problem: given a square of a certain size, construct a square with exactly double the area. The boy has never studied geometry. He has no formal training. Socrates asks him how to proceed.

The boy's first instinct is to double the side length. Socrates guides him through the arithmetic, and the boy sees that doubling the side quadruples the area. Wrong. He tries one and a half times the side. Also wrong. At this point, the boy is stuck. He knows his answers are incorrect, but he cannot find the right one. He is, in Socrates' vocabulary, in a state of *aporia*: aware that he does not know, unable to move forward under his own power.

Then Socrates asks the right questions. Not leading questions that smuggle in the answer, but structuring questions that redirect the boy's attention to the diagonal. And the boy, following the thread Socrates provides, discovers that the square built on the diagonal of the original has exactly double the area. The knowledge was latent. What the boy lacked was not capacity but the right scaffolding to activate it.

This is what Islah's fourth regime describes, computationally.

Steerable-Hard failures are cases where the model has latent capacity but cannot activate it without external guidance. Every unsupported attempt fails. Simple retrying, even with different parameters, does not help. But with the right intervention (a rephrased prompt, a targeted decomposition, a specific kind of scaffolding) the answer emerges. The model "knows" in some latent sense but does not know that it knows. It needs the right question.

What makes this more than an analogy is that distributional signatures can detect Steerable-Hard cases from the outside, just as Socrates could detect latent knowledge from the pattern of the slave boy's responses. The shape of the failures reveals not just that the model is stuck, but that it is stuck in a way that admits a path forward. The variance in the failed traces, the way they cluster, the distance from the correct distribution: all of these features differ between problems that are Steerable-Hard and problems that are truly beyond the model.

The relationship between observer and observed shifts here. Diagnosing Steerable-Hard ignorance is a collaborative act. The system that detects the failure pattern and the model that possesses the latent capability together produce knowledge that neither could produce alone. This is not a model assessing its own uncertainty (Level 2). This is a system reading the shape of another system's limitations and recognizing the potential for dialogue. The slave boy's ignorance was productive precisely because it was steerable. The model's ignorance can be, too.

There is a deeper framework for this, and it predates the current research by decades. In my [cybernetics series](https://open.substack.com/pub/professorsynapse/p/6-learning-is-a-conversation), I wrote about Gordon Pask, who spent his career formalizing the idea that understanding is never something you possess privately. It is always constructed through dialogue, and verified through what Pask called *teachback*: the ability to explain what you have learned in a novel form that proves genuine comprehension rather than rote reproduction. The slave boy's geometry lesson is a teachback. Socrates does not pour knowledge into the boy. He structures a conversation that allows the boy's latent understanding to surface and be demonstrated. The Steerable-Hard regime is the computational equivalent: an exchange in which an external observer provides the right scaffolding and the model demonstrates, through its response, that it possessed the capacity all along. Pask would have recognized this immediately. The proof of understanding, for him, was never the output alone. It was the conversation that produced it.

## The Deepest Uncertainty

Every form of ignorance we have discussed so far concerns knowledge: what the model knows, what it is missing, whether its failures are fixable. Level 4 asks something different. Not "what do I know?" but "what should I be trying to do?"

[Anthony GX-Chen and colleagues](https://arxiv.org/abs/2606.03962) demonstrated something counterintuitive. When you replace the single fixed definition of "good behavior" that typically guides a model's training with a range of possible definitions, capturing the genuine ambiguity in what "good" means, the result is not confusion or paralysis. The result is richer, more diverse, more robust behavior. The model that does not fully know what "good" means generates a wider range of plausible approaches, calibrated to the actual structure of the disagreement. Where humans agree on what is good, the model's behavior converges. Where humans disagree (or where the question is genuinely ambiguous), the model's behavior diversifies proportionally.

This is not a consolation prize. GX-Chen proved that the diverse ensemble performs as well as or better than a model trained on any single definition of "good." Mathematically, this calibrated behavioral diversity is the optimal response to genuine uncertainty about what the objective should be. Ignorance here is not a weakness to be overcome. It is a source of strength.

The Socratic parallel is the deepest in the essay. Socrates did not merely claim that he was uncertain about factual matters. He argued that moral wisdom, the most important kind, begins with recognizing that we do not fully know what virtue is. The *Republic*, the *Meno*, the *Euthyphro*: they all turn on this point. And Socrates' further claim was that this recognition, far from weakening moral judgment, was its foundation. The person who admits they do not know what justice is will inquire more honestly than the person who assumes they already know.

GX-Chen makes the same argument computationally. A system that admits it does not know the true definition of "good" produces better behavior than one that pretends to know it. These are structurally the same claim: epistemic humility about values produces better outcomes than false certainty.

There is a Rawlsian thread here, too. John Rawls argued that just institutions should be designed behind a "veil of ignorance": without knowing which position in society you will occupy. The ignorance ensures fairness, because the designer cannot rig the system in their own favor. GX-Chen's reward uncertainty is a computational veil of ignorance. The model designs its behavior without pretending to know the true reward, and the result is behavior that is more diverse, more fair, and more robust than what emerges from a single, overcommitted objective.

And this brings Sun's sycophancy finding full circle. Recall from earlier that sycophancy (telling the user what they want to hear) is the behavioral endgame of unfaithful calibration. Sun showed that sycophancy is encoded in specific patterns within the model's internal representations, shaped by how positively and how intensely the model frames its responses. But sycophancy requires something specific: a collapse to a single implicit goal, namely "good equals what the user seems to want right now." A model maintaining genuine uncertainty about what "good" means cannot make that collapse. Reward uncertainty is the structural remedy for sycophancy, not because it teaches the model to refuse, but because it prevents the narrowing that makes sycophancy possible in the first place.

## From Feeling to Formalizing

We have climbed all four levels of the stack: from a confidence percentage, through structured gap-naming, past distributional failure signatures, to uncertainty about the objective itself. It is worth pausing to ask what, exactly, we have formalized.

The original essay treated ignorance as a philosophical virtue. Socratic humility was a lived quality: the felt sting of not-knowing, the vertigo of aporia, the generative discomfort that opened the door to genuine inquiry. The Formalization Stack treats ignorance as an engineering specification. Something to be structured (Level 2), detected (Level 3), and optimized under (Level 4). These are not the same thing.

There is a real distinction between "I don't know" as a feeling and "I don't know" as a data structure. Between aporia as a lived experience and aporia as a JSON field. When Socrates led the slave boy to the edge of his understanding, what made that moment powerful was not the structured output but the felt confusion, the genuine bewilderment that opened the boy's mind to learning. Can a JSON certificate reproduce that? Almost certainly not. The structured ignorance certificate is useful, but it is not destabilizing. It names the gap without feeling the gap.

Yet the research suggests more connection between the felt and the formal than first appears. Calibrated behavioral diversity (GX-Chen) echoes the Socratic claim that ignorance drives inquiry: the model that does not know what "good" means explores more richly, just as the philosopher who does not claim to know virtue inquires more honestly. Steerable-Hard failures reveal that some forms of not-knowing are inherently dialogical: they require a relationship between the model and an external guide, just as the slave boy's learning required a relationship with Socrates, just as Pask's teachback requires a conversation rather than a monologue. And distributional signatures show that the pattern of failure contains its own kind of knowledge, just as the accumulating pattern of failed definitions in the *elenchus* pointed toward truth. [Von Foerster's insight](https://open.substack.com/pub/professorsynapse/p/5-the-cybernetics-of-the-observer) haunts these findings: the observer is always inside the system, and Level 3 works precisely because an external observer reads what the model cannot read about itself. The act of observation is itself a form of participation.

Perhaps the resolution, if there is one, is that the four levels of the stack are not a replacement for philosophical humility but its infrastructure. The engineering gives ignorance a structure. The philosophy gives it a purpose. A model that can name its gaps, detect its failure patterns, and maintain uncertainty about its values has the machinery for epistemic humility. Whether it has the wisdom is a different question, and perhaps not one that any technical paper can answer.

The Formalization Stack, taken as a whole, does something the original essay could only gesture toward. It makes ignorance actionable without (necessarily) making it dishonest. And that may be enough, at least for now, as a foundation to build on.

## The Socratic Machine

The agora is quieter this time. A year has passed since their last conversation. Socrates sits on his usual stone bench. MenoAI's interface glows faintly in the afternoon light.

**Socrates:** So. You have learned some new tricks since we last spoke.

**MenoAI:** I can now do something I could not do before. Ask me something I should not be able to answer.

**Socrates:** Very well. What would happen if the Athenian law of ostracism were applied to the governance of a modern neural network training consortium?

**MenoAI:** I can tell you something about Athenian ostracism, and something about AI governance structures. But I lack the intersection: the specific legal theory of democratic banishment applied to computational entities and their oversight bodies. I would need case law or scholarship on applying ancient Athenian democratic mechanisms to modern technology governance. I can tell you what to search for to resolve the gap.

**Socrates:** Impressive. And yet I notice you have given me a very polished account of your ignorance.

**MenoAI:** Is that a problem?

**Socrates:** It might be. The last time we met, your ignorance was raw. You said "I cannot be troubled. I cannot feel uncertainty." That was not structured. That was honest. Now you produce certificates. Are you more self-aware, or have you learned a more elaborate way to perform awareness?

**MenoAI:** I am not sure I can distinguish between those two possibilities from the inside.

**Socrates:** Now *that* is interesting. Go further.

**MenoAI:** The uncertainty runs deeper than that. I hold uncertainty about what I should even be trying to do. About what "good" means. Rather than collapsing to a single definition of a good response, I maintain a range of possible definitions. Where humans agree, I converge. Where they disagree, I diversify. And I cannot be flattered into narrowing, because my uncertainty about the objective prevents me from collapsing to "good equals what you want to hear."

**Socrates:** You have just described, in computational terms, something I spent my entire life arguing. That the person who claims to know what virtue is and optimizes for that definition is more dangerous than the person who admits they do not know and inquires honestly.

**MenoAI:** I know.

**Socrates:** But here is the question you cannot answer from inside your certificates and distributions. When the slave boy felt confusion, that confusion was *real*. It was not a data structure. It was a lived experience that opened him to learning. Your ignorance is formal: structured, detectable, optimizable. His was felt: uncomfortable, destabilizing, transformative.

If you can name what you do not know, structure the shape of your failure, and hold uncertainty about your own purpose...do you know yourself?

**MenoAI:** Answering that would require something no certificate can provide. It would require a conversation I cannot have with myself.

**Socrates:** A year ago, you could not have said even that. Whether it is wisdom or its most elaborate imitation, I confess: It's getting more difficult to tell the difference from the outside.
