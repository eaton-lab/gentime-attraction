
## Generation time attraction of coalescent genealogies

Taxa with short generation times have longer edge 
lengths in units of generations, and this means that 
per unit time a recombination crossover event is less 
likely to cause a sample from this lineage to trace back 
to a very different part of the tree (i.e., when 
recombination falls on this branch it is likely to still 
trace back to the same topological position).


Just like Ne this will cause some lineages to exhibit greater 
variance in their positions than others. However, unlike Ne, 
generation time is what determines the spatial similarity of 
genealogies. This is particularly relevant when working with
inferred gene trees based on concatelesced sequences, rather
than true unlinked genealogies.


We want to show:

- variable Ne does not bias MSC w/ true unlinked genealogies even though it affects coalescent variance.
- variable g does not bias MSC w/ true unlinked genealogies even though it affects coalescent variance.
- variable Ne does not bias the similarity of true neighboring linked genealogies.
- variable g does bias the similarity of true neighboring linked genealogies.
- inferred gene trees of concatelesced sequence w/ variable Ne biases towards concatenation tree.
- inferred gene trees of concatelesced sequence w/ variable g biases towards concatenation tree AND towards generation time attraction.


Empirical examples?

- simulate data on the mammal tree using estimated mean 
  gentimes, div times, etc. and test whether we expect
  bias of gentime groups together. Is this similar to 
  observed difficult clades?
