package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.clustering.ClusterParameters;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.SilhouetteEvaluation;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.DatasetType;
import com.gengoai.apollo.ml.data.format.CSVDataFormat;
import com.gengoai.apollo.ml.data.format.DataFormat;
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
   private final ClusterParameters fitParameters;

   public BaseClustererTest(Clusterer clusterer, ClusterParameters fitParameters) {
      this.clusterer = clusterer;
      this.fitParameters = fitParameters;
   }


   public Clusterer convertClustering(Clusterer clustering) {
      return clustering;
   }

   @Test
   public void fitAndEvaluate() {
      SilhouetteEvaluation evaluation = new SilhouetteEvaluation(fitParameters.measure);
      evaluation.evaluate(convertClustering(clusterer.fit(loadWaterData(), fitParameters)));
      assertTrue(passes(evaluation));
   }

   protected Dataset loadWaterData() {
      DataFormat csv = new CSVDataFormat(CSV.builder(), -1);
      try {
         return DatasetType.InMemory.read(Resources.fromClasspath("com/gengoai/apollo/ml/water-treatment.data"),
                                          csv);
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }


   protected abstract boolean passes(SilhouetteEvaluation evaluation);

}//END OF BaseClustererTest
