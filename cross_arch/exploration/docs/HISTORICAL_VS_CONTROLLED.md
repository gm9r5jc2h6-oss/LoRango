# Historical exploration versus controlled exploration

The SD3 and FLUX experiments were reached through iterative manual work. That
history is evidence that the experiments were performed, but it is not by
itself a prospective algorithm for a new architecture.

This bundle therefore keeps two claims separate.

## Historical reconstruction

The neighboring fixed reproducibility capsule records the concrete SD3
E17-04A and FLUX F32 topology, tensors, losses, training order, and export. It
answers: “How can this named construction be rebuilt?”

Historical run names or failed experiments may be reported in a trace, but
they are not silently converted into a deterministic search rule after the
fact.

## Controlled prospective policy

`controlled_v1` answers: “Given a new architecture, what fixed process will we
apply before observing its validation results?” It specifies:

- candidate templates;
- ranks and initialization seeds;
- probe updates and total budget;
- four-condition metrics;
- scalar score and thresholds;
- tie-breaking;
- maximum selected candidates;
- a negative stopping outcome.

The policy is an auditable porting procedure. It is not claimed to reproduce
every historical manual choice.

## Wording supported by the release

> The reported SD3 and FLUX examples are reconstructed with fixed
> architecture-specific configurations. Separately, we provide a prospective
> controlled exploration policy for porting the mechanism to a new
> architecture. The policy records all candidates and permits a negative
> outcome; it is not presented as a literal replay of the historical manual
> search.

