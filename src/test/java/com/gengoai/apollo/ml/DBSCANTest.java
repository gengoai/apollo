package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.DBSCAN;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.stat.measure.Distance;

/**
 * @author David B. Bracewell
 */
public class DBSCANTest extends BaseClustererTest {

   public DBSCANTest() {
      super(new DBSCAN(),
            new DBSCAN.Parameters()
               .set("eps", 1e-10)
               .set("distanceMeasure", Distance.Manhattan));
   }

   @Override
   public boolean passes(SilhouetteEvaluation mce) {
      mce.output();
      return mce.getAvgSilhouette() >= 0.9;
   }


}//END OF DBSCANTest
