package com.davidbracewell.apollo.ml.classification;

/**
 * @author David B. Bracewell
 */
public class SoftmaxTest extends BaseClassificationTest {
   public SoftmaxTest() {
      super(ClassifierLearner.classification()
                             .learnerClass(SoftmaxLearner.class)
                             .parameter("verbose", false)
                             .build(),
            0.94,
            0.05);
   }
}//END OF SGDTest
