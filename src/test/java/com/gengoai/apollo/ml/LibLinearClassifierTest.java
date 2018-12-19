package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.LibLinearModel;
import com.gengoai.apollo.ml.classification.MultiClassEvaluation;
import com.gengoai.apollo.ml.classification.PipelinedClassifier;
import com.gengoai.apollo.ml.vectorizer.IndexVectorizer;

/**
 * @author David B. Bracewell
 */
public class LibLinearClassifierTest extends BaseClassifierTest {

   public LibLinearClassifierTest() {
      super(new PipelinedClassifier(new LibLinearModel(), IndexVectorizer.featureVectorizer()),
            new LibLinearModel.Parameters());
   }

   @Override
   public boolean passes(MultiClassEvaluation mce) {
      return mce.microF1() >= 0.85;
   }
}//END OF LibLinearClassifierTest
