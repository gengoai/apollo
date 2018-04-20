package com.gengoai.apollo.ml.classification;

/**
 * @author David B. Bracewell
 */
public class BinaryGradientDescentTest extends BaseClassificationTest {
   public BinaryGradientDescentTest() {
      super(BinaryGradientDescentLearner.logisticRegression()
                                        .setParameter("verbose", false)
                                        .oneVsRest(), 0.94, 0.1);
   }
}//END OF SGDTest
