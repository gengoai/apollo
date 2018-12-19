package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.LinearModel;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.apollo.ml.classification.PipelinedClassifier;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;

/**
 * @author David B. Bracewell
 */
public class LinearModelTest extends BaseClassifierTest {

   public LinearModelTest() {
      super(new PipelinedClassifier(new LinearModel(), IndexVectorizer.featureVectorizer()),
            new LinearModel.Parameters()
               .set("verbose", false)
               .set("maxIterations", 20));
   }

   @Override
   public boolean passes(MultiClassEvaluation mce) {
      return mce.microF1() >= 0.85;
   }
}//END OF LinearModelTest
