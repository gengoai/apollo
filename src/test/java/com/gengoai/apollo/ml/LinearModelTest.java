package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.ClassifierEvaluation;
import com.gengoai.apollo.ml.classification.LinearModel;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;
import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class LinearModelTest extends BaseClassifierTest {

   public LinearModelTest() {
      super(new LinearModel(IndexVectorizer.labelVectorizer(),
                            IndexVectorizer.featureVectorizer()),
            new LinearModel.Parameters()
               .set("verbose", false)
               .set("maxIterations", 100));
   }

   @Override
   public boolean passes(ClassifierEvaluation evaluation) {
      MultiClassEvaluation mce = Cast.as(evaluation);
      mce.output();
      return mce.microF1() >= 0.85;
   }
}//END OF LinearModelTest
