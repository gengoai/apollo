package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.DBSCAN;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.ml.preprocess.FilterPreprocessor;

/**
 * @author David B. Bracewell
 */
public class DBSCANTest extends BaseClustererTest {

   public DBSCANTest() {
      super(new DBSCAN(new FilterPreprocessor("AutoColumn_0")),
            DBSCAN.eps.set(500.0));
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      return mce.getAvgSilhouette() >= 0.9;
   }


}//END OF DBSCANTest
