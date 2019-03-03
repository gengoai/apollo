package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.ClassifierEvaluation;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.apollo.ml.classification.NaiveBayes;
import com.gengoai.apollo.ml.preprocess.PerFeatureTransform;
import com.gengoai.apollo.ml.preprocess.ZScoreTransform;
import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class NaiveBayesTest extends BaseClassifierTest {

   public NaiveBayesTest() {
      super(new NaiveBayes(new PerFeatureTransform(ZScoreTransform::new)));
   }

   @Override
   public boolean passes(ClassifierEvaluation evaluation) {
      MultiClassEvaluation mce = Cast.as(evaluation);
      return mce.microF1() >= 0.28;
   }
}//END OF NaiveBayesTest
