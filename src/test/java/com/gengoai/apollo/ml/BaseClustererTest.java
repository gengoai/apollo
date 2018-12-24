package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.ml.data.CSVDataSource;
import com.gengoai.apollo.ml.data.DataSource;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.DatasetType;
import com.gengoai.io.CSV;
import com.gengoai.io.Resources;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class BaseClustererTest {
   private final Clusterer clusterer;
   private final FitParameters fitParameters;

   public BaseClustererTest(Clusterer clusterer, FitParameters fitParameters) {
      this.clusterer = clusterer;
      this.fitParameters = fitParameters;
   }


   @Test
   public void fitAndEvaluate() {
      clusterer.fit(loadWaterData(), fitParameters);
      SilhouetteEvaluation evaluation = new SilhouetteEvaluation();
      evaluation.evaluate(clusterer, loadWaterData());
      assertTrue(passes(evaluation));
   }

   protected Dataset loadWaterData() {
      DataSource csv = new CSVDataSource(CSV.builder(), -1);
      try {
         return DatasetType.InMemory.read(Resources.fromClasspath("com/gengoai/apollo/ml/water-treatment.data"),
                                          csv);
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }


   protected abstract boolean passes(SilhouetteEvaluation evaluation);

}//END OF BaseClustererTest
