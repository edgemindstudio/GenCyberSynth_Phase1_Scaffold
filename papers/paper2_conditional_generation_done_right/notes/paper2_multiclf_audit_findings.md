# Paper 2 Multi-Classifier External Audit Findings



## Purpose



This note records the Phase 2 journal-upgrade experiment for Paper 2. The goal was to test whether the fake-class-head ACGAN conditioning failure was an artifact of a single external audit classifier.



## Audit classifiers



We evaluated the b100 fake-class-head ACGAN synthetic manifests using two additional real-only audit classifiers:



1. ResNet-style CNN audit classifier

2. Linear SVM-style flattened-image audit classifier



The original CNN audit classifier remains the baseline audit model.



## Real-test performance



- ResNet-style CNN real-test accuracy: approximately 0.9604

- Linear SVM-flat real-test accuracy: approximately 0.9167



Both audit classifiers achieved strong performance on held-out real data.



## Results



### ResNet-style CNN



- Seed 42: conditioning accuracy 0.1200, leakage 0.8800, predicted-label collapse mainly into class 4, with a small class-8 component.

- Seed 43: conditioning accuracy 0.1111, leakage 0.8889, full collapse to class 2.

- Seed 44: conditioning accuracy 0.1111, leakage 0.8889, full collapse to class 4.



### Linear SVM-flat



- Seed 42: conditioning accuracy 0.1111, leakage 0.8889, full collapse to class 2.

- Seed 43: conditioning accuracy 0.1256, leakage 0.8744, collapse into classes 2, 4, 5, and 7.

- Seed 44: conditioning accuracy 0.1111, leakage 0.8889, full collapse to class 2.



## Interpretation



The exact predicted collapse class varies across audit classifiers, but all classifiers report severe external conditioning failure. This supports the claim that the fake-class-head ACGAN failure is not an artifact of a single audit classifier.



The important result is not that every audit classifier predicts the same collapse class. Rather, the important result is that none of the independent audit classifiers recognize the generated samples as their requested labels at high rates. Across classifiers, generated samples are assigned to one class or a small subset of classes instead of preserving the requested class distribution.



## Journal-paper implication



This result strengthens Paper 2 by addressing a likely reviewer concern: that external conditioning audit outcomes may depend on the choice of audit model. The evidence shows that the central finding is robust across multiple independent real-only audit classifiers.
