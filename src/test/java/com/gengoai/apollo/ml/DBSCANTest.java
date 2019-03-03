package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.DBSCAN;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.ml.preprocess.FilterPreprocessor;
import com.gengoai.conversion.Cast;

/**
 * @author David B. Bracewell
 */
public class DBSCANTest extends BaseClustererTest<DBSCAN.Parameters> {

   public DBSCANTest() {
      super(new DBSCAN(new FilterPreprocessor("AutoColumn_0")),
            Cast.as(new DBSCAN.Parameters()
                       .set("eps", 500)));
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      return mce.getAvgSilhouette() >= 0.9;
   }


}//END OF DBSCANTest
